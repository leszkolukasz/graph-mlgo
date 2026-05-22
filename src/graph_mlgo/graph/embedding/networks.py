from abc import ABC
from typing import Callable

import jax.numpy as jnp
from flax import linen as nn

from graph_mlgo.graph.embedding.aggregator import Aggregator


class EmbeddingNet(ABC, nn.Module):
    pass


class GraphSageNet(EmbeddingNet):
    depth: int
    hidden_dim: int
    output_dim: int
    aggregator_cls: type[Aggregator]
    activation: Callable = nn.sigmoid

    def setup(self):
        self.input_proj = nn.Dense(self.hidden_dim, name="input_proj")
        self.edge_embedder = nn.Embed(num_embeddings=4, features=self.hidden_dim)

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
    # edge_types: list of (num_targets, num_neighbours)
    # Returns: (N, output_dim)
    def __call__(
        self,
        h: jnp.ndarray,
        neighbor_indices: list[jnp.ndarray],
        edge_types: list[jnp.ndarray],
    ) -> jnp.ndarray:

        h = self.input_proj(h)

        for d in range(self.depth):
            num_targets = neighbor_indices[d].shape[0]

            h_target = h[:num_targets]
            h_neighbors = h[neighbor_indices[d]]

            e_emb = self.edge_embedder(edge_types[d])
            h_neighbors = h_neighbors + e_emb

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


class GATNet(EmbeddingNet):
    depth: int
    hidden_dim: int
    output_dim: int
    num_heads: int = 4
    ffn_scale: int | None = 4

    pad_id: int = -1

    def setup(self):
        self.input_proj = nn.Dense(self.hidden_dim, name="input_proj")

        self.edge_embedder = nn.Embed(num_embeddings=4, features=self.hidden_dim)

        self.attentions = [
            nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                out_features=self.hidden_dim if d < self.depth - 1 else self.output_dim,
                broadcast_dropout=False,
                name=f"gat_attn_{d}",
            )
            for d in range(self.depth)
        ]

        self.attn_norms = [
            nn.LayerNorm(name=f"attn_norm_{d}") for d in range(self.depth)
        ]

        if self.ffn_scale is not None:
            ffn_hidden_dim = self.hidden_dim * self.ffn_scale
            self.ffn_norms = [
                nn.LayerNorm(name=f"ffn_norm_{d}") for d in range(self.depth - 1)
            ]
            self.ffn_dense1 = [
                nn.Dense(ffn_hidden_dim, name=f"ffn_d1_{d}")
                for d in range(self.depth - 1)
            ]
            self.ffn_dense2 = [
                nn.Dense(self.hidden_dim, name=f"ffn_d2_{d}")
                for d in range(self.depth - 1)
            ]

    def __call__(
        self,
        h: jnp.ndarray,
        neighbor_indices: list[jnp.ndarray],
        edge_types: list[jnp.ndarray],
    ) -> jnp.ndarray:
        h = self.input_proj(h)

        for d in range(self.depth):
            num_nodes = h.shape[0]

            h_norm = self.attn_norms[d](h)

            h_q = jnp.expand_dims(h_norm, axis=1)  # (N, 1, features)
            h_neighbors = h_norm[neighbor_indices[d]]  # (N, num_neighbours, features)

            e_emb = self.edge_embedder(edge_types[d])
            h_neighbors = h_neighbors + e_emb

            h_kv = jnp.concatenate(
                [h_q, h_neighbors], axis=1
            )  # (N, 1 + num_neighbours, features)

            self_mask = jnp.ones((num_nodes, 1), dtype=jnp.bool_)
            neighbors_mask = neighbor_indices[d] != self.pad_id
            mask_2d = jnp.concatenate([self_mask, neighbors_mask], axis=1)

            # (batch, num_heads, q_len, kv_len)
            # (N, 1, 1, 1 + num_neighbours)
            attention_mask = jnp.expand_dims(mask_2d, axis=(1, 2))

            h_attn = self.attentions[d](
                inputs_q=h_q, inputs_kv=h_kv, mask=attention_mask
            )  # (N, 1, out_features)
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
