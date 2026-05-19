from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import VariableDict

from graph_mlgo.constants import MAX_NODES
from graph_mlgo.graph.embedding.aggregator import Aggregator
from graph_mlgo.graph.embedding.config import GraphSageConfig
from graph_mlgo.graph.embedding.constants import GLOBAL_FEATURES_DIM, NODE_FEATURES_DIM
from graph_mlgo.graph.embedding.utils import extract_neighborhood

if TYPE_CHECKING:
    from graph_mlgo.graph import Edge, Graph


class Embedder(ABC):
    def embed(self, edge: "Edge", graph: "Graph") -> np.ndarray:
        global_feat = graph.get_global_features()
        edge_embed = self._embed(edge, graph)
        edge_mult = np.array([graph.edges[edge]], dtype=np.float32)
        edge_mult = np.log1p(edge_mult)

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

        return np.concatenate([global_feat, edge_embed, edge_mult, const_ratio])

    def get_embedding_dim(self) -> int:
        return self._get_embedding_dim() + GLOBAL_FEATURES_DIM + 2

    @abstractmethod
    def _embed(self, edge: "Edge", graph: "Graph") -> np.ndarray:
        pass

    @abstractmethod
    def _get_embedding_dim(self) -> int:
        pass


class TrivialEmbedder(Embedder):
    def _embed(self, edge: "Edge", graph: "Graph") -> np.ndarray:
        u, v = edge
        feat_u = graph.nodes[u].features
        feat_v = graph.nodes[v].features

        return np.concatenate([feat_u, feat_v])

    def _get_embedding_dim(self) -> int:
        return 2 * NODE_FEATURES_DIM


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
        self.config = config

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

        current_num_nodes = feat_np.shape[0]

        if current_num_nodes > MAX_NODES:
            raise ValueError(
                f"Subgraph exceeds MAX_NODES: {current_num_nodes} > {MAX_NODES}"
            )

        pad_size = MAX_NODES - current_num_nodes

        feat_np_padded = np.pad(feat_np, ((0, pad_size), (0, 0)), mode="constant")

        indices_np_padded = []
        for ind in indices_np:
            num_targets = ind.shape[0]
            pad_targets = MAX_NODES - num_targets

            ind_padded = np.pad(
                ind, ((0, pad_targets), (0, 0)), mode="constant", constant_values=0
            )
            indices_np_padded.append(ind_padded)

        feat_jax, indices_jax = (
            jnp.asarray(feat_np_padded, dtype=jnp.float32),
            [jnp.asarray(ind, dtype=jnp.int32) for ind in indices_np_padded],
        )

        emb = np.asarray(self._jit_apply(feat_jax, indices_jax))
        # assert emb.shape[0] == 2

        return np.concatenate([feat_u, feat_v, emb[0], emb[1]])

    def _get_embedding_dim(self) -> int:
        return 2 * self.net.output_dim + 2 * NODE_FEATURES_DIM

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
    ) -> "GraphSAGEEmbedder":
        from graph_mlgo.graph.embedding.training.trainer import GraphSAGETrainer

        trainer, runner_state, _ = GraphSAGETrainer.load(checkpoint_path)

        return cls(
            net=trainer.model,
            params=runner_state.train_state.params,
            config=trainer.config,
        )
