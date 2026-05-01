import jax.numpy as jnp
from typing import NamedTuple

class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray


class RolloutInfo(NamedTuple):
    completed_return: jnp.ndarray
    completed_length: jnp.ndarray
    completed: jnp.ndarray


class RunningNorm(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


class RunnerState(NamedTuple):
    train_state: object
    obs: jnp.ndarray
    rng: jnp.ndarray
    obs_norm: RunningNorm
    rew_norm: RunningNorm
    running_returns: jnp.ndarray
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
