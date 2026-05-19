from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import linen as nn
from flax.typing import VariableDict
from loguru import logger

from graph_mlgo.graph.embedding.aggregator import NAME_TO_CLASS, Aggregator
from graph_mlgo.graph.embedding.config import GraphSageConfig
from graph_mlgo.graph.embedding.utils import extract_neighborhood

if TYPE_CHECKING:
    from graph_mlgo.graph import Edge, Graph


class Embedder(ABC):
    def embed(self, edge: "Edge", graph: "Graph") -> np.ndarray:
        global_feat = graph.get_global_features()
        node_feat = self._embed(edge, graph)
        edge_mult = np.array([graph.edges[edge]], dtype=np.float32)

        caller, callee = edge
        total_args = 0
        const_args = 0

        caller_func = graph.module.get_function(caller)
        for block in caller_func.blocks:
            for instr in block.instructions:
                if instr.opcode in ("call", "invoke"):
                    ops = list(instr.ope10000rands)
                    if ops and getattr(ops[-1], "name", "") == callee:
                        args = ops[:-1]
                        total_args += len(args)
                        const_args += sum(
                            1 for arg in args if getattr(arg, "is_constant", False)
                        )

        const_ratio = np.array(
            [float(const_args) / total_args if total_args > 0 else 0.0],
            dtype=np.float32,
        )

        return np.concatenate([global_feat, node_feat, edge_mult, const_ratio])

    def get_embedding_dim(self, node_feat_dim: int, global_feat_dim: int) -> int:
        return self._get_embedding_dim(node_feat_dim) + global_feat_dim + 2

    @abstractmethod
    def _embed(self, edge: "Edge", graph: "Graph") -> np.ndarray:
        pass

    @abstractmethod
    def _get_embedding_dim(self, node_feat_dim: int) -> int:
        pass


class TrivialEmbedder(Embedder):
    def _embed(self, edge: "Edge", graph: "Graph") -> np.ndarray:
        u, v = edge
        feat_u = graph.nodes[u].features
        feat_v = graph.nodes[v].features

        return np.concatenate([feat_u, feat_v])

    def _get_embedding_dim(self, node_feat_dim: int) -> int:
        return 2 * node_feat_dim


class GraphSAGENet(nn.Module):
    depth: int
    hidden_dim: int
    output_dim: int
    aggregator_cls: type[Aggregator]
    activation: Callable = nn.sigmoid

    def setup(self):
        self.W = [
            nn.Dense(
                self.output_dim if d == self.depth - 1 else self.hidden_dim,
                name=f"W_{d}",
            )
            for d in range(self.depth)
        ]

        self.aggregators = [
            self.aggregator_cls(name=f"agg_{d}") for d in range(self.depth)
        ]

        self.norms = [nn.LayerNorm(name=f"norm_{d}") for d in range(self.depth - 1)]

    # h: (N, node_feat_dim)
    # neighbor_indices: list of (num_targets, num_neighbours, hidden_dim)
    # Returns: (N, output_dim)
    def __call__(
        self, h: jnp.ndarray, neighbor_indices: list[jnp.ndarray]
    ) -> jnp.ndarray:

        # TODO: scan
        for d in range(self.depth):
            num_targets = neighbor_indices[d].shape[0]

            h_target = h[:num_targets]
            h_neighbors = h[neighbor_indices[d]]

            h_aggregated = self.aggregators[d](h_neighbors)
            h_concat = jnp.concatenate([h_target, h_aggregated], axis=-1)

            h_new = self.W[d](h_concat)

            if d < self.depth - 1:
                h_new = self.norms[d](h_new)
                h_new = self.activation(h_new)

            h_new = h_new / jnp.maximum(
                jnp.linalg.norm(h_new, axis=-1, keepdims=True), 1e-6
            )

            h = h_new

        return h


class GraphSAGEEmbedder(Embedder):
    net: GraphSAGENet
    params: VariableDict
    config: GraphSageConfig

    def __init__(
        self,
        *,
        net: GraphSAGENet,
        params: VariableDict,
        config: GraphSageConfig,
    ):
        self.net = net
        self.params = params

        def _apply(h_in: jnp.ndarray, neigh_indices: list[jnp.ndarray]) -> jnp.ndarray:
            return self.net.apply(self.params, h_in, neigh_indices)  # ty: ignore

        apply_fn_type = Callable[[jnp.ndarray, list[jnp.ndarray]], jnp.ndarray]
        self._jit_apply = cast(apply_fn_type, jax.jit(_apply))

    def _embed(self, edge: "Edge", graph: "Graph") -> np.ndarray:
        u, v = edge

        feat_u = graph.nodes[u].features
        feat_v = graph.nodes[v].features

        feat_np, indices_np = extract_neighborhood(
            graph=graph,
            batch=[u, v],
            depth=self.config.depth,
            num_neighbours=self.config.num_neighbours,
        )
        feat_jax, indices_jax = (
            jnp.asarray(feat_np, dtype=jnp.float32),
            [jnp.asarray(ind, dtype=jnp.int32) for ind in indices_np],
        )

        emb = np.asarray(self._jit_apply(feat_jax, indices_jax))
        assert emb.shape[0] == 2

        return np.concatenate([feat_u, feat_v, emb[0], emb[1]])

    def _get_embedding_dim(self, node_feat_dim: int) -> int:
        return 2 * self.net.output_dim + 2 * node_feat_dim

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        node_feat_dim: int,
    ) -> "GraphSAGEEmbedder":
        cp_path = Path(checkpoint_path)

        config_path = cp_path / "config.yaml"
        config = GraphSageConfig.from_file(config_path)

        logger.info(f"Loaded GraphSAGE config: {config}")

        net = GraphSAGENet(
            depth=config.depth,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            aggregator_cls=NAME_TO_CLASS[config.aggregator_type],
        )

        rng = jax.random.PRNGKey(config.seed)
        dummy_h = jnp.zeros((1, node_feat_dim))
        dummy_indices = [
            jnp.zeros((1, config.num_neighbours), dtype=jnp.int32)
            for _ in range(config.depth)
        ]

        variables = net.init(rng, dummy_h, dummy_indices)
        empty_params = variables["params"]

        mngr = ocp.CheckpointManager(checkpoint_path)
        latest_step = mngr.latest_step()

        if latest_step is None:
            raise FileNotFoundError(f"Checkpoint not found in {checkpoint_path}")

        restored_params = mngr.restore(
            latest_step, args=ocp.args.PyTreeRestore(item=empty_params)
        )

        return cls(net=net, params=restored_params, config=config)
