import jax
import jax.numpy as jnp
from graph_mlgo.agent.utils import normalize
from graph_mlgo.agent.config import PPOConfig

class PPOEvaluator:
    def __init__(self, config: PPOConfig, eval_env, agent):
        self.config = config
        self.eval_env = eval_env
        self.agent = agent

    def make_eval_fn(self):
        cfg = self.config
        eval_env = self.eval_env
        agent = self.agent

        def evaluate(train_state, obs_norm, rng: jax.Array):
            rng, reset_rng = jax.random.split(rng)
            env_state = eval_env.reset(reset_rng)
            obs = env_state.obs
            episode_returns = jnp.zeros((obs.shape[0],), dtype=jnp.float32)
            total_completed_return = jnp.array(0.0, dtype=jnp.float32)
            total_completed_count = jnp.array(0.0, dtype=jnp.float32)

            def eval_step(carry, _):
                (
                    env_state,
                    obs,
                    episode_returns,
                    total_completed_return,
                    total_completed_count,
                    rng,
                ) = carry

                obs_in = (
                    normalize(obs, obs_norm, eps=cfg.norm_eps, clip=cfg.obs_clip)
                    if cfg.normalize_obs
                    else obs
                )
                rng, act_rng = jax.random.split(rng)
                _, action, _ = agent.act(train_state.params, obs_in, act_rng)

                next_env_state = eval_env.step(env_state, action)
                reward = next_env_state.reward
                done = next_env_state.done.astype(jnp.float32)
                next_obs = next_env_state.obs

                episode_returns = episode_returns + reward
                completed_return = jnp.where(done > 0.0, episode_returns, 0.0)
                total_completed_return = total_completed_return + jnp.sum(
                    completed_return
                )
                total_completed_count = total_completed_count + jnp.sum(done)

                episode_returns = episode_returns * (1.0 - done)

                next_carry = (
                    next_env_state,
                    next_obs,
                    episode_returns,
                    total_completed_return,
                    total_completed_count,
                    rng,
                )
                return next_carry, None

            final_carry, _ = jax.lax.scan(
                eval_step,
                (
                    env_state,
                    obs,
                    episode_returns,
                    total_completed_return,
                    total_completed_count,
                    rng,
                ),
                None,
                length=cfg.eval_horizon,
            )
            _, _, _, total_completed_return, total_completed_count, _ = final_carry

            mean_return = total_completed_return / jnp.maximum(
                total_completed_count, 1.0
            )
            return {
                "eval_return": mean_return,
                "eval_episodes": total_completed_count,
            }

        return jax.jit(evaluate)