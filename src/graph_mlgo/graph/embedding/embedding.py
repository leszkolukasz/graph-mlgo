from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, cast

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.typing import VariableDict

from graph_mlgo.constants import MAX_NODES
from graph_mlgo.graph.embedding.aggregator import Aggregator
from graph_mlgo.graph.embedding.config import EmbeddingConfig
from graph_mlgo.graph.embedding.constants import GLOBAL_FEATURES_DIM, NODE_FEATURES_DIM
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
    config: EmbeddingConfig

    pad_id: int = -1

    def __init__(
        self,
        *,
        net: GraphSAGENet,
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

        # feat_u = graph.nodes[u].features
        # feat_v = graph.nodes[v].features

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

        feat_np_padded = jnp.pad(
            feat_np,
            ((0, pad_size), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

        indices_np_padded = []
        for ind in indices_np:
            num_targets = ind.shape[0]
            pad_targets = MAX_NODES - num_targets

            ind_padded = jnp.pad(
                ind,
                ((0, pad_targets), (0, 0)),
                mode="constant",
                constant_values=self.pad_id,
            )
            indices_np_padded.append(ind_padded)

        feat_jax, indices_jax = (
            jnp.asarray(feat_np_padded, dtype=jnp.float32),
            [jnp.asarray(ind, dtype=jnp.int32) for ind in indices_np_padded],
        )

        emb = jnp.asarray(self._jit_apply(self.params, feat_jax, indices_jax))
        # assert emb.shape[0] == 2

        return jnp.concatenate([emb[0], emb[1]]), EmbeddingAux(
            h=feat_jax, indices=indices_jax
        )

    def _get_embedding_dim(self) -> int:
        return 2 * self.net.output_dim

    @classmethod
    def load(cls, *, checkpoint_path: str, rng: jax.Array) -> "GraphSAGEEmbedder":
        from graph_mlgo.graph.embedding.training.trainer import GraphSAGETrainer

        trainer, runner_state, _ = GraphSAGETrainer.load(
            rng=rng, checkpoint_path=checkpoint_path
        )

        return cls(
            net=trainer.model,
            params=runner_state.train_state.params,
            config=trainer.config,
        )

class GATNet(nn.Module):
    depth: int
    hidden_dim: int
    output_dim: int
    num_heads: int = 4
    ffn_scale: int | None = 4
    
    pad_id: int = -1

    def setup(self):
        self.attentions = [
            nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                out_features=self.hidden_dim if d < self.depth - 1 else self.output_dim,
                broadcast_dropout=False,
                name=f"gat_attn_{d}",
            )
            for d in range(self.depth)
        ]

        self.attn_norms = [nn.LayerNorm(name=f"attn_norm_{d}") for d in range(self.depth)]

        if self.ffn_scale is not None:
            ffn_hidden_dim = self.hidden_dim * self.ffn_scale
            self.ffn_norms = [nn.LayerNorm(name=f"ffn_norm_{d}") for d in range(self.depth - 1)]
            self.ffn_dense1 = [nn.Dense(ffn_hidden_dim, name=f"ffn_d1_{d}") for d in range(self.depth - 1)]
            self.ffn_dense2 = [nn.Dense(self.hidden_dim, name=f"ffn_d2_{d}") for d in range(self.depth - 1)]

    def __call__(
        self, h: jnp.ndarray, neighbor_indices: list[jnp.ndarray]
    ) -> jnp.ndarray:

        for d in range(self.depth):
            num_nodes = h.shape[0]

            h_norm = self.attn_norms[d](h)

            h_q = jnp.expand_dims(h_norm, axis=1)  # (N, 1, features)
            h_neighbors = h_norm[neighbor_indices[d]]  # (N, num_neighbours, features)
            h_kv = jnp.concatenate([h_q, h_neighbors], axis=1) # (N, 1 + num_neighbours, features)

            self_mask = jnp.ones((num_nodes, 1), dtype=jnp.bool_)
            neighbors_mask = neighbor_indices[d] != self.pad_id
            mask_2d = jnp.concatenate([self_mask, neighbors_mask], axis=1)

            # (batch, num_heads, q_len, kv_len)
            # (N, 1, 1, 1 + num_neighbours)
            attention_mask = jnp.expand_dims(mask_2d, axis=(1, 2))

            h_attn = self.attentions[d](
                inputs_q=h_q, inputs_kv=h_kv, mask=attention_mask
            ) # (N, 1, out_features)
            h_attn = jnp.squeeze(h_attn, axis=1)

            if d < self.depth - 1:
                h = h + h_attn

                if self.ffn_scale is not None:
                    h_ffn = self.ffn_norms[d](h)
                    
                    h_ffn = self.ffn_dense1[d](h_ffn)
                    h_ffn = nn.gelu(h_ffn)
                    h_ffn = self.ffn_dense2[d](h_ffn)
                    
                    h = h + h_ffn
            else:
                h = h_attn

        return h

class GATEmbedder(Embedder):
    net: GATNet
    params: VariableDict
    config: Any  # Podmień na np. GATConfig, jeśli używasz oddzielnego obiektu konfiguracji

    pad_id: int = -1

    def __init__(
        self,
        *,
        net: GATNet,
        params: VariableDict,
        config: Any,
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
            pad_id=self.pad_id
        )

        feat_jax = jnp.asarray(feat_np_padded, dtype=jnp.float32)
        indices_jax = [jnp.asarray(ind, dtype=jnp.int32) for ind in indices_np_padded]

        emb = jnp.asarray(self._jit_apply(self.params, feat_jax, indices_jax))

        return jnp.concatenate([emb[0], emb[1]]), EmbeddingAux(
            h=feat_jax, indices=indices_jax
        )

    def _get_embedding_dim(self) -> int:
        return 2 * self.net.output_dim

    @classmethod
    def load(cls, *, checkpoint_path: str, rng: jax.Array) -> "GATEmbedder":
        from graph_mlgo.graph.embedding.training.trainer import GATTrainer

        trainer, runner_state, _ = GATTrainer.load(
            rng=rng, checkpoint_path=checkpoint_path
        )

        return cls(
            net=trainer.model,
            params=runner_state.train_state.params,
            config=trainer.config,
        )