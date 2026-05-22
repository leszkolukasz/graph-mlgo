from pathlib import Path
from typing import Callable, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from loguru import logger

from graph_mlgo.graph import Graph
from graph_mlgo.graph.embedding import GATNet, GraphSageNet
from graph_mlgo.graph.embedding.aggregator import NAME_TO_CLASS
from graph_mlgo.graph.embedding.config import EmbeddingConfig
from graph_mlgo.graph.embedding.constants import NODE_FEATURES_DIM
from graph_mlgo.graph.embedding.networks import EmbeddingNet
from graph_mlgo.graph.embedding.utils import contrastive_loss, extract_neighborhood


class EmbeddingRunnerState(NamedTuple):
    train_state: TrainState


class EmbeddingTrainer:
    config: EmbeddingConfig
    model: EmbeddingNet
    mngr: ocp.CheckpointManager

    def __init__(self, *, model: EmbeddingNet, config: EmbeddingConfig):
        self.model = model
        self.config = config

        options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
        self.mngr = ocp.CheckpointManager(config.checkpoint_dir, options=options)

    @classmethod
    def load(
        cls,
        *,
        rng: jax.Array,
        checkpoint_path: str | None = None,
        config: EmbeddingConfig | None = None,
    ) -> tuple["EmbeddingTrainer", EmbeddingRunnerState, int]:
        assert checkpoint_path is not None or config is not None, (
            "Must provide either checkpoint_path or config to load the trainer."
        )
        checkpoint_path = checkpoint_path or config.checkpoint_dir  # ty: ignore
        cp_path = Path(checkpoint_path).absolute()

        config_path = cp_path / "config.yaml"
        if config is None:
            config = EmbeddingConfig.from_file(config_path)
        elif config_path.exists():
            config_from_file = EmbeddingConfig.from_file(config_path)
            if config_from_file != config:
                raise ValueError(
                    f"Config provided does not match config in checkpoint. Provided: {config}, Checkpoint config: {config_from_file}"
                )

        model = (
            GraphSageNet(
                depth=config.depth,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                aggregator_cls=NAME_TO_CLASS[config.aggregator_type],
            )
            if config.embedding_type == "graphsage"
            else GATNet(
                depth=config.depth,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                num_heads=config.num_heads,
            )
        )

        trainer = cast("EmbeddingTrainer", cls(model=model, config=config))
        runner_state = trainer.init_runner(rng)

        mngr = ocp.CheckpointManager(cp_path)
        latest_step = mngr.latest_step()

        if latest_step is not None:
            runner_state = cast(
                EmbeddingRunnerState,
                mngr.restore(
                    latest_step, args=ocp.args.PyTreeRestore(item=runner_state)
                ),
            )
            start_update = latest_step + 1
            logger.warning(
                f"Found existing embedding trainer checkpoint at {cp_path}, step: {latest_step}."
            )
        else:
            start_update = 0

        return trainer, runner_state, start_update

    def save_checkpoint(
        self, runner_state: EmbeddingRunnerState, update_idx: int
    ) -> None:
        self.config.save()

        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            runner_state,
        )
        self.mngr.save(update_idx, args=orbax.checkpoint.args.PyTreeSave(runner_state))

    def init_runner(self, rng: jax.Array) -> EmbeddingRunnerState:
        dummy_h = jnp.zeros((1, NODE_FEATURES_DIM))

        dummy_indices = [
            jnp.zeros((1, self.config.num_neighbours), dtype=jnp.int32)
            for _ in range(self.model.depth)
        ]

        dummy_parts = jnp.zeros((1,), dtype=jnp.int32)

        params = self.model.init(rng, dummy_h, dummy_indices, dummy_parts)

        tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.lr, eps=1e-5),
        )

        train_state = TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

        return EmbeddingRunnerState(
            train_state=train_state,
        )

    make_update_fn_ret_type = Callable[
        [EmbeddingRunnerState, tuple[list[str], list[str], list[list[str]]], Graph],
        tuple[EmbeddingRunnerState, dict[str, jnp.ndarray]],
    ]

    def make_update_fn(self) -> make_update_fn_ret_type:
        model = self.model

        def _jax_update(
            train_state: TrainState,
            h_in: jnp.ndarray,
            neighbor_indices: list[jnp.ndarray],
            edge_types: list[jnp.ndarray],
            u_idx: jnp.ndarray,
            v_idx: jnp.ndarray,
            neg_idx: jnp.ndarray,
        ) -> tuple[TrainState, dict[str, jnp.ndarray]]:

            def loss_fn(params: VariableDict):
                return contrastive_loss(
                    params,
                    model,
                    h_in,
                    neighbor_indices,
                    edge_types,
                    u_idx,
                    v_idx,
                    neg_idx,
                )

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, metrics), grads = grad_fn(train_state.params)

            new_train_state = train_state.apply_gradients(grads=grads)

            return new_train_state, metrics

        jax_update_typ = Callable[
            [
                TrainState,
                jnp.ndarray,
                list[jnp.ndarray],
                list[jnp.ndarray],
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
            ],
            tuple[TrainState, dict[str, jnp.ndarray]],
        ]
        jax_update = cast(jax_update_typ, jax.jit(_jax_update))

        def update_batch(
            runner_state: EmbeddingRunnerState,
            batch_data: tuple[list[str], list[str], list[list[str]]],
            graph: Graph,
        ) -> tuple[EmbeddingRunnerState, dict[str, jnp.ndarray]]:
            batch_u, batch_v, batch_neg = batch_data

            all_target_nodes = list(batch_u) + list(batch_v)
            for neg_list in batch_neg:
                all_target_nodes.extend(neg_list)

            h_np, neighbor_indices_np, edge_types_np = extract_neighborhood(
                graph=graph,
                batch=all_target_nodes,
                depth=self.config.depth,
                num_neighbours=self.config.num_neighbours,
                use_in_edges=self.config.use_in_edges,
            )

            B = len(batch_u)
            Q = len(batch_neg[0])

            u_idx = np.arange(0, B, dtype=np.int32)
            v_idx = np.arange(B, 2 * B, dtype=np.int32)
            neg_idx = np.arange(2 * B, 2 * B + B * Q, dtype=np.int32).reshape((B, Q))

            h_jax = jnp.asarray(a=h_np, dtype=jnp.float32)
            neighbor_indices_jax = [
                jnp.asarray(idx, dtype=jnp.int32) for idx in neighbor_indices_np
            ]
            edge_types_jax = [jnp.asarray(et, dtype=jnp.int32) for et in edge_types_np]
            u_idx_jax = jnp.asarray(u_idx)
            v_idx_jax = jnp.asarray(v_idx)
            neg_idx_jax = jnp.asarray(neg_idx)

            next_train_state, metrics = jax_update(
                runner_state.train_state,
                h_jax,
                neighbor_indices_jax,
                edge_types_jax,
                u_idx_jax,
                v_idx_jax,
                neg_idx_jax,
            )

            next_runner_state = EmbeddingRunnerState(
                train_state=next_train_state,
            )

            return next_runner_state, metrics

        return update_batch
