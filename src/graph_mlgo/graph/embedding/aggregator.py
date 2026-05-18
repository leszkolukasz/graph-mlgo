from abc import ABC

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
