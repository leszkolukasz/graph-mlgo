from typing import NamedTuple

import jax.numpy as jnp

from graph_mlgo.env.LLVMInline import Observation


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


class PPORunnerState(NamedTuple):
    train_state: dict
    obs: Observation
    rng: jnp.ndarray
    obs_norm: RunningNorm
    rew_norm: RunningNorm
    running_returns: jnp.ndarray
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
