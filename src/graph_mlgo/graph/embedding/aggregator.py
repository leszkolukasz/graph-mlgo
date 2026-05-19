from abc import ABC
from typing import Callable

import jax.numpy as jnp
from flax import linen as nn


class Aggregator(nn.Module, ABC):
    pass


class MeanAggregator(Aggregator):
    # h_neighbors: (num_targets, num_neighbors, hidden_dim)
    # Returns: (num_targets, out_dim)
    @nn.compact
    def __call__(self, h_neighbors: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(h_neighbors, axis=-2)


class PoolAggregator(Aggregator):
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, h_neighbors: jnp.ndarray) -> jnp.ndarray:
        hidden_dim = h_neighbors.shape[-1]

        h_transformed = nn.Dense(features=hidden_dim, name="pool_dense")(h_neighbors)

        h_transformed = self.activation(h_transformed)

        return jnp.max(h_transformed, axis=-2)


NAME_TO_CLASS = {
    "mean": MeanAggregator,
    "pool": PoolAggregator,
}
