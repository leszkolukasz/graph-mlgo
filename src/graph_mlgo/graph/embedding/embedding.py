from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import VariableDict

from graph_mlgo.graph.embedding.aggregator import Aggregator
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
                    ops = list(instr.operands)
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
    activation = nn.relu
    W: list[nn.Dense]
    aggregator_cls: type[Aggregator]
    aggregators: list[Aggregator]

    def __init__(
        self,
        depth: int,
        hidden_dim: int,
        output_dim: int,
        aggregator_cls: type[Aggregator],
    ):
        super().__init__()
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.aggregator_cls = aggregator_cls

    def setup(self):
        self.W = []
        for d in range(self.depth):
            out_dim = self.output_dim if d == self.depth - 1 else self.hidden_dim
            self.W.append(nn.Dense(out_dim))

        self.aggregators = []
        for _ in range(self.depth):
            self.aggregators.append(self.aggregator_cls())

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
            h_new = self.activation(h_new)  # ty: ignore

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

        return np.concatenate([emb[0], emb[1]])

    def _get_embedding_dim(self, node_feat_dim: int) -> int:
        return 2 * self.net.output_dim
