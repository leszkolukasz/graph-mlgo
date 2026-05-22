from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, cast

import jax
import jax.numpy as jnp
from flax.typing import VariableDict

from graph_mlgo.constants import MAX_NODES
from graph_mlgo.graph.embedding.config import EmbeddingConfig
from graph_mlgo.graph.embedding.constants import GLOBAL_FEATURES_DIM, NODE_FEATURES_DIM
from graph_mlgo.graph.embedding.networks import EmbeddingNet, GATNet, GraphSageNet
from graph_mlgo.graph.embedding.utils import (
    EmbeddingAux,
    EmbeddingParts,
    concatenate_parts,
    extract_neighborhood,
    pad_neighborhood,
)

if TYPE_CHECKING:
    from graph_mlgo.graph import Edge, Graph


class Embedder(ABC):
    def embed(self, edge: "Edge", graph: "Graph") -> tuple[jnp.ndarray, EmbeddingParts]:
        parts = self._get_embedding_parts(edge, graph)
        return concatenate_parts(parts), parts

    def _get_embedding_parts(self, edge: "Edge", graph: "Graph") -> EmbeddingParts:
        global_feat = graph.get_global_features()
        edge_embed, aux = self._embed(edge, graph)
        edge_mult = jnp.array([graph.edges[edge]], dtype=jnp.float32)
        edge_mult = jnp.log1p(edge_mult)

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

        const_ratio = jnp.array(
            [float(const_args) / total_args if total_args > 0 else 0.0],
            dtype=jnp.float32,
        )

        return EmbeddingParts(
            global_feat=jnp.array(global_feat, dtype=jnp.float32),
            edge_embed=jnp.array(edge_embed, dtype=jnp.float32),
            edge_mult=jnp.array(edge_mult, dtype=jnp.float32),
            const_ratio=jnp.array(const_ratio, dtype=jnp.float32),
            aux=aux,
        )

    def get_embedding_dim(self) -> int:
        return self._get_embedding_dim() + GLOBAL_FEATURES_DIM + 2

    @abstractmethod
    def _embed(
        self, edge: "Edge", graph: "Graph"
    ) -> tuple[jnp.ndarray, EmbeddingAux | None]:
        pass

    @abstractmethod
    def _get_embedding_dim(self) -> int:
        pass


class TrivialEmbedder(Embedder):
    def _embed(
        self, edge: "Edge", graph: "Graph"
    ) -> tuple[jnp.ndarray, EmbeddingAux | None]:
        u, v = edge
        feat_u = graph.nodes[u].features
        feat_v = graph.nodes[v].features

        return jnp.concatenate([feat_u, feat_v]), None

    def _get_embedding_dim(self) -> int:
        return 2 * NODE_FEATURES_DIM


class NetEmbedder(Embedder):
    net: EmbeddingNet
    params: VariableDict
    config: EmbeddingConfig

    pad_id: int = -1

    def __init__(
        self,
        *,
        net: EmbeddingNet,
        params: VariableDict,
        config: EmbeddingConfig,
    ):
        self.net = net
        self.params = params
        self.config = config

        def _apply(
            params: VariableDict,
            h_in: jnp.ndarray,
            neigh_indices: list[jnp.ndarray],
            edge_types: list[jnp.ndarray],
        ) -> jnp.ndarray:
            return self.net.apply(params, h_in, neigh_indices, edge_types)  # ty: ignore

        apply_fn_type = Callable[
            [VariableDict, jnp.ndarray, list[jnp.ndarray], list[jnp.ndarray]],
            jnp.ndarray,
        ]
        self._jit_apply = cast(apply_fn_type, jax.jit(_apply))

    def _embed(
        self, edge: "Edge", graph: "Graph"
    ) -> tuple[jnp.ndarray, EmbeddingAux | None]:
        u, v = edge

        feat_np, indices_np, edge_types_np = extract_neighborhood(
            graph=graph,
            batch=[u, v],
            depth=self.config.depth,
            num_neighbours=self.config.num_neighbours,
            use_in_edges=self.config.use_in_edges,
        )

        feat_np_padded, indices_np_padded, edge_types_np_padded = pad_neighborhood(
            feat_np=feat_np,
            indices_np=indices_np,
            edge_types_np=edge_types_np,
            max_nodes=MAX_NODES,
            pad_id=self.pad_id,
        )

        feat_jax = jnp.asarray(feat_np_padded, dtype=jnp.float32)
        indices_jax = [jnp.asarray(ind, dtype=jnp.int32) for ind in indices_np_padded]
        edge_types_jax = [
            jnp.asarray(et, dtype=jnp.int32) for et in edge_types_np_padded
        ]

        emb = jnp.asarray(
            self._jit_apply(self.params, feat_jax, indices_jax, edge_types_jax)
        )

        return jnp.concatenate([emb[0], emb[1]]), EmbeddingAux(
            h=feat_jax, indices=indices_jax, edge_types=edge_types_jax
        )

    def _get_embedding_dim(self) -> int:
        return 2 * self.net.output_dim


class GraphSageEmbedder(NetEmbedder):
    net: GraphSageNet

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: str | None = None,
        config: EmbeddingConfig | None = None,
        rng: jax.Array,
    ) -> "NetEmbedder":
        from graph_mlgo.graph.embedding.training.trainer import EmbeddingTrainer

        trainer, runner_state, _ = EmbeddingTrainer.load(
            rng=rng, checkpoint_path=checkpoint_path, config=config
        )

        assert isinstance(trainer.model, GraphSageNet)

        return cls(
            net=trainer.model,
            params=runner_state.train_state.params,
            config=trainer.config,
        )


class GATEmbedder(NetEmbedder):
    net: GATNet

    @classmethod
    def load(
        cls,
        *,
        checkpoint_path: str | None = None,
        config: EmbeddingConfig | None = None,
        rng: jax.Array,
    ) -> "GATEmbedder":
        from graph_mlgo.graph.embedding.training.trainer import EmbeddingTrainer

        trainer, runner_state, _ = EmbeddingTrainer.load(
            rng=rng, checkpoint_path=checkpoint_path, config=config
        )

        assert isinstance(trainer.model, GATNet)

        return cls(
            net=trainer.model,
            params=runner_state.train_state.params,
            config=trainer.config,
        )


EMBEDDER_TYPE_MAP = {
    "graphsage": GraphSageEmbedder,
    "gat": GATEmbedder,
}
