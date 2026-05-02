import jax
import jax.numpy as jnp
import numpy as np
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


        @jax.jit
        def jax_act(params, obs_norm, obs, rng):
            obs_in = (
                normalize(obs, obs_norm, eps=cfg.norm_eps, clip=cfg.obs_clip)
                if cfg.normalize_obs
                else obs
            )
            rng, act_rng = jax.random.split(rng)
            
            action, _, _ = agent.act(params, obs_in, act_rng)
            
            return action, rng

        def evaluate(train_state, obs_norm, rng: jax.Array):
            obs, _ = eval_env.reset()

            episode_returns = np.zeros(obs.shape[0], dtype=np.float32)
            total_completed_return = 0.0
            total_completed_count = 0.0

            for _ in range(cfg.eval_horizon):
                obs_jax = jnp.asarray(obs, dtype=jnp.float32)
                action_jax, rng = jax_act(train_state.params, obs_norm, obs_jax, rng)
                
                action_np = np.asarray(action_jax)
            
                next_obs, reward, terminated, truncated, _ = eval_env.step(action_np)    
                done = np.logical_or(terminated, truncated).astype(np.float32)

                episode_returns += reward
                
                completed_return = np.where(done > 0.0, episode_returns, 0.0)
                total_completed_return += float(np.sum(completed_return))
                total_completed_count += float(np.sum(done))

                episode_returns *= (1.0 - done)
                obs = next_obs

            mean_return = total_completed_return / max(total_completed_count, 1.0)
            
            return {
                "eval_return": float(mean_return),
                "eval_episodes": float(total_completed_count),
            }

        return evaluate