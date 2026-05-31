from typing import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import VariableDict

from graph_mlgo.agent.config import PPOConfig
from graph_mlgo.agent.networks import PPOAgent
from graph_mlgo.agent.training.types import RunningNorm
from graph_mlgo.agent.utils import normalize


class PPOEvaluator:
    config: PPOConfig
    eval_env: gym.Env
    agent: PPOAgent

    def __init__(self, config: PPOConfig, eval_env: gym.Env, agent: PPOAgent):
        self.config = config
        self.eval_env = eval_env
        self.agent = agent

    make_eval_fn_ret_type = Callable[
        [VariableDict, RunningNorm, jax.Array], dict[str, float]
    ]

    def make_eval_fn(self) -> make_eval_fn_ret_type:
        cfg = self.config
        eval_env = self.eval_env
        agent = self.agent

        @jax.jit
        def jax_act(params, obs_norm, obs, rng):
            obs_in = (
                normalize(obs, obs_norm, eps=cfg.norm_eps, clip=cfg.obs_clip)
                if cfg.normalize_obs
                else obs
            )
            rng, act_rng = jax.random.split(rng)

            _, determ_action, _ = agent.act(params, obs_in, act_rng)

            return determ_action, rng

        def evaluate(
            train_state: VariableDict, obs_norm: RunningNorm, rng: jax.Array
        ) -> dict[str, float]:
            obs, _ = eval_env.reset()

            episode_returns = np.zeros(obs.embedding.shape[0], dtype=np.float32)
            total_completed_return = 0.0
            total_completed_count = 0.0

            for _ in range(cfg.eval_horizon):
                obs_jax = jnp.asarray(obs.embedding, dtype=jnp.float32)
                action_jax, rng = jax_act(train_state.params, obs_norm, obs_jax, rng)  # ty: ignore

                next_obs, reward, terminated, truncated, _ = eval_env.step(action_jax)
                done = np.logical_or(terminated, truncated).astype(np.float32)

                episode_returns += reward

                completed_return = np.where(done > 0.0, episode_returns, 0.0)
                total_completed_return += float(np.sum(completed_return))
                total_completed_count += float(np.sum(done))

                episode_returns *= 1.0 - done
                obs = next_obs

            mean_return = total_completed_return / max(total_completed_count, 1.0)

            return {
                "eval_return": float(mean_return),
                "eval_episodes": float(total_completed_count),
            }

        return evaluate
