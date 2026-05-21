from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, cast

import jax
import jax.numpy as jnp
from flax import struct
from flax.typing import VariableDict

from graph_mlgo.constants import MAX_NODES
from graph_mlgo.graph.embedding.config import EmbeddingConfig
from graph_mlgo.graph.embedding.constants import GLOBAL_FEATURES_DIM, NODE_FEATURES_DIM
from graph_mlgo.graph.embedding.networks import EmbeddingNet, GATNet, GraphSageNet
from graph_mlgo.graph.embedding.utils import extract_neighborhood, pad_neighborhood

if TYPE_CHECKING:
    from graph_mlgo.graph import Edge, Graph


@struct.dataclass
class EmbeddingAux:
    h: jnp.ndarray
    indices: list[jnp.ndarray]

    def to_device(self, device: jax.Device) -> "EmbeddingAux":
        return EmbeddingAux(
            h=jax.device_put(self.h, device),
            indices=[jax.device_put(idx, device) for idx in self.indices],
        )

    def to_cpu(self) -> "EmbeddingAux":
        return self.to_device(jax.devices("cpu")[0])

    def to_gpu(self) -> "EmbeddingAux":
        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("No GPU devices available for EmbeddingAux")
        return self.to_device(gpus[0])


@struct.dataclass
class EmbeddingParts:
    global_feat: jnp.ndarray
    edge_embed: jnp.ndarray
    edge_mult: jnp.ndarray
    const_ratio: jnp.ndarray
    aux: EmbeddingAux | None = None

    def to_device(self, device: jax.Device) -> "EmbeddingParts":
        return EmbeddingParts(
            global_feat=jax.device_put(self.global_feat, device),
            edge_embed=jax.device_put(self.edge_embed, device),
            edge_mult=jax.device_put(self.edge_mult, device),
            const_ratio=jax.device_put(self.const_ratio, device),
            aux=self.aux.to_device(device) if self.aux is not None else None,
        )

    def to_cpu(self) -> "EmbeddingParts":
        return self.to_device(jax.devices("cpu")[0])

    def to_gpu(self) -> "EmbeddingParts":
        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("No GPU devices available for EmbeddingParts")
        return self.to_device(gpus[0])

    @classmethod
    def empty(cls, embed_dim: int) -> "EmbeddingParts":
        return cls(
            global_feat=jnp.zeros(GLOBAL_FEATURES_DIM, dtype=jnp.float32),
            edge_embed=jnp.zeros(embed_dim, dtype=jnp.float32),
            edge_mult=jnp.zeros(1, dtype=jnp.float32),
            const_ratio=jnp.zeros(1, dtype=jnp.float32),
            aux=None,
        )


class Embedder(ABC):
    def embed(self, edge: "Edge", graph: "Graph") -> tuple[jnp.ndarray, EmbeddingParts]:
        parts = self._get_embedding_parts(edge, graph)
        return self._concatenate_parts(parts), parts

    def _concatenate_parts(self, parts: EmbeddingParts) -> jnp.ndarray:
        return jnp.concatenate(
            [parts.global_feat, parts.edge_embed, parts.edge_mult, parts.const_ratio]
        )

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
            params: VariableDict, h_in: jnp.ndarray, neigh_indices: list[jnp.ndarray]
        ) -> jnp.ndarray:
            return self.net.apply(params, h_in, neigh_indices)  # ty: ignore

        apply_fn_type = Callable[
            [VariableDict, jnp.ndarray, list[jnp.ndarray]], jnp.ndarray
        ]
        self._jit_apply = cast(apply_fn_type, jax.jit(_apply))

    def _embed(
        self, edge: "Edge", graph: "Graph"
    ) -> tuple[jnp.ndarray, EmbeddingAux | None]:
        u, v = edge

        feat_np, indices_np = extract_neighborhood(
            graph=graph,
            batch=[u, v],
            depth=self.config.depth,
            num_neighbours=self.config.num_neighbours,
        )

        feat_np_padded, indices_np_padded = pad_neighborhood(
            feat_np=feat_np,
            indices_np=indices_np,
            max_nodes=MAX_NODES,
            pad_id=self.pad_id,
        )

        feat_jax = jnp.asarray(feat_np_padded, dtype=jnp.float32)
        indices_jax = [jnp.asarray(ind, dtype=jnp.int32) for ind in indices_np_padded]

        emb = jnp.asarray(self._jit_apply(self.params, feat_jax, indices_jax))

        return jnp.concatenate([emb[0], emb[1]]), EmbeddingAux(
            h=feat_jax, indices=indices_jax
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
