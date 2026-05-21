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

from graph_mlgo.graph import Graph
from graph_mlgo.graph.embedding import GraphSAGENet
from graph_mlgo.graph.embedding.aggregator import NAME_TO_CLASS
from graph_mlgo.graph.embedding.config import EmbeddingConfig
from graph_mlgo.graph.embedding.constants import NODE_FEATURES_DIM
from graph_mlgo.graph.embedding.utils import extract_neighborhood, graphsage_loss


class GraphSAGERunnerState(NamedTuple):
    train_state: TrainState


class GraphSAGETrainer:
    config: EmbeddingConfig
    model: GraphSAGENet
    mngr: ocp.CheckpointManager

    def __init__(self, *, model: GraphSAGENet, config: EmbeddingConfig):
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
    ) -> tuple["GraphSAGETrainer", GraphSAGERunnerState, int]:
        assert checkpoint_path is not None or config is not None, (
            "Must provide either checkpoint_path or config to load GraphSAGETrainer"
        )
        checkpoint_path = checkpoint_path or config.checkpoint_dir  # ty: ignore
        cp_path = Path(checkpoint_path).absolute()

        if config is None:
            config_path = cp_path / "config.yaml"
            config = EmbeddingConfig.from_file(config_path)

        model = GraphSAGENet(
            depth=config.depth,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            aggregator_cls=NAME_TO_CLASS[config.aggregator_type],
        )

        trainer = cast("GraphSAGETrainer", cls(model=model, config=config))
        runner_state = trainer.init_runner(rng)

        mngr = ocp.CheckpointManager(cp_path)
        latest_step = mngr.latest_step()

        if latest_step is not None:
            runner_state = cast(
                GraphSAGERunnerState,
                mngr.restore(
                    latest_step, args=ocp.args.PyTreeRestore(item=runner_state)
                ),
            )
            start_update = latest_step + 1
        else:
            start_update = 0

        return trainer, runner_state, start_update

    def save_checkpoint(
        self, runner_state: GraphSAGERunnerState, update_idx: int
    ) -> None:
        self.config.save()

        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            runner_state,
        )
        self.mngr.save(update_idx, args=orbax.checkpoint.args.PyTreeSave(runner_state))

    def init_runner(self, rng: jax.Array) -> GraphSAGERunnerState:
        dummy_h = jnp.zeros((1, NODE_FEATURES_DIM))

        dummy_indices = [
            jnp.zeros((1, self.config.num_neighbours), dtype=jnp.int32)
            for _ in range(self.model.depth)
        ]

        params = self.model.init(rng, dummy_h, dummy_indices)

        tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.lr, eps=1e-5),
        )

        train_state = TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

        return GraphSAGERunnerState(
            train_state=train_state,
        )

    make_update_fn_ret_type = Callable[
        [GraphSAGERunnerState, tuple[list[str], list[str], list[list[str]]], Graph],
        tuple[GraphSAGERunnerState, dict[str, jnp.ndarray]],
    ]

    def make_update_fn(self) -> make_update_fn_ret_type:
        model = self.model

        def _jax_update(
            train_state: TrainState,
            h_in: jnp.ndarray,
            neighbor_indices: list[jnp.ndarray],
            u_idx: jnp.ndarray,
            v_idx: jnp.ndarray,
            neg_idx: jnp.ndarray,
        ) -> tuple[TrainState, dict[str, jnp.ndarray]]:

            def loss_fn(params: VariableDict):
                return graphsage_loss(
                    params, model, h_in, neighbor_indices, u_idx, v_idx, neg_idx
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
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
            ],
            tuple[TrainState, dict[str, jnp.ndarray]],
        ]
        jax_update = cast(jax_update_typ, jax.jit(_jax_update))

        def update_batch(
            runner_state: GraphSAGERunnerState,
            batch_data: tuple[list[str], list[str], list[list[str]]],
            graph: Graph,
        ) -> tuple[GraphSAGERunnerState, dict[str, jnp.ndarray]]:
            batch_u, batch_v, batch_neg = batch_data

            all_target_nodes = list(batch_u) + list(batch_v)
            for neg_list in batch_neg:
                all_target_nodes.extend(neg_list)

            h_np, neighbor_indices_np = extract_neighborhood(
                graph=graph,
                batch=all_target_nodes,
                depth=self.config.depth,
                num_neighbours=self.config.num_neighbours,
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
            u_idx_jax = jnp.asarray(u_idx)
            v_idx_jax = jnp.asarray(v_idx)
            neg_idx_jax = jnp.asarray(neg_idx)

            next_train_state, metrics = jax_update(
                runner_state.train_state,
                h_jax,
                neighbor_indices_jax,
                u_idx_jax,
                v_idx_jax,
                neg_idx_jax,
            )

            next_runner_state = GraphSAGERunnerState(
                train_state=next_train_state,
            )

            return next_runner_state, metrics

        return update_batch
