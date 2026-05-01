import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import gymnasium as gym

from .utils import compute_gae, ppo_loss, init_running_norm, update_running_norm, normalize
from .types import Transition, RolloutInfo, RunnerState
from .config import PPOConfig
from .networks import PPOAgent


class PPOTrainer:
    
    def __init__(self, config: PPOConfig, env: gym.Env, agent: PPOAgent):
        self.config = config
        self.env = env
        self.agent = agent

    def init_runner(self, rng: jax.Array):
        rng, init_rng = jax.random.split(rng, 2)
        obs, _ = self.env.reset()
        obs_dim = obs.shape[-1]

        params = self.agent.init(init_rng, jnp.zeros((1, obs_dim)))

        def learning_rate_schedule(step):
            frac = 1.0 - step / (
                self.config.num_updates
                * self.config.update_epochs
                * self.config.num_minibatches
            )
            return self.config.lr * frac

        tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(
                learning_rate_schedule if self.config.anneal_lr else self.config.lr,
                eps=1e-5,
            ),
        )

        train_state = TrainState.create(apply_fn=self.agent.apply, params=params, tx=tx)

        obs_norm = init_running_norm((obs_dim,))
        rew_norm = init_running_norm(())
        running_returns = jnp.zeros((self.config.num_envs,), dtype=jnp.float32)
        episode_returns = jnp.zeros((self.config.num_envs,), dtype=jnp.float32)
        episode_lengths = jnp.zeros((self.config.num_envs,), dtype=jnp.int32)

        return RunnerState(
            train_state=train_state,
            obs=obs,
            rng=rng,
            obs_norm=obs_norm,
            rew_norm=rew_norm,
            running_returns=running_returns,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
        )

    def make_update_fn(self):
        cfg = self.config
        env = self.env
        agent = self.agent

        def env_step(carry, _):
            runner_state = carry

            if cfg.normalize_obs:
                obs_norm = update_running_norm(runner_state.obs_norm, runner_state.obs)
                obs_in = normalize(
                    runner_state.obs, obs_norm, eps=cfg.norm_eps, clip=cfg.obs_clip
                )
            else:
                obs_norm = runner_state.obs_norm
                obs_in = runner_state.obs

            rng, act_rng = jax.random.split(runner_state.rng)
            action, _, log_prob = agent.act(
                runner_state.train_state.params, obs_in, act_rng
            )
            value = agent.apply(
                runner_state.train_state.params, obs_in, method=agent.critic
            )

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = jnp.array(next_obs, dtype=jnp.float32)
            reward = jnp.array(reward, dtype=jnp.float32)
            done = jnp.logical_or(terminated, truncated).astype(jnp.float32)

            return_for_norm = runner_state.running_returns * cfg.gamma + reward
            if cfg.normalize_reward:
                rew_norm = update_running_norm(runner_state.rew_norm, reward)
                reward_out = reward / jnp.sqrt(jnp.maximum(rew_norm.var, cfg.norm_eps))
                reward_out = jnp.clip(reward_out, -cfg.reward_clip, cfg.reward_clip)
            else:
                rew_norm = runner_state.rew_norm
                reward_out = reward
            next_running_returns = return_for_norm * (1.0 - done)

            new_episode_returns = runner_state.episode_returns + reward
            new_episode_lengths = runner_state.episode_lengths + 1
            completed_return = jnp.where(done > 0.0, new_episode_returns, 0.0)
            completed_length = jnp.where(done > 0.0, new_episode_lengths, 0)

            episode_returns = new_episode_returns * (1.0 - done)
            episode_lengths = (new_episode_lengths * (1.0 - done)).astype(jnp.int32)

            transition = Transition(
                obs=obs_in,
                action=action,
                reward=reward_out,
                done=done,
                value=value, # ty: ignore
                log_prob=log_prob,
            )
            rollout_info = RolloutInfo(
                completed_return=completed_return,
                completed_length=completed_length,
                completed=done,
            )
            next_carry = RunnerState(
                train_state=runner_state.train_state,
                obs=next_obs,
                rng=rng,
                obs_norm=obs_norm,
                rew_norm=rew_norm,
                running_returns=next_running_returns,
                episode_returns=episode_returns,
                episode_lengths=episode_lengths,
            )
            return next_carry, (transition, rollout_info)

        def update_minibatch(train_state, minibatch):
            batch, adv_batch, target_batch = minibatch

            def loss_on_params(params):
                return ppo_loss(params, agent, batch, adv_batch, target_batch, cfg)

            grad_fn = jax.value_and_grad(loss_on_params, has_aux=True)
            (loss, aux), grads = grad_fn(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, (loss, aux)

        def update_epoch(carry, _):
            train_state, flat_traj, flat_adv, flat_targets, rng = carry

            rng, perm_rng = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, cfg.batch_size)

            shuffled_traj = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), flat_traj
            )
            shuffled_adv = jnp.take(flat_adv, permutation, axis=0)
            shuffled_targets = jnp.take(flat_targets, permutation, axis=0)

            minibatches = (
                jax.tree_util.tree_map(
                    lambda x: x.reshape(
                        (cfg.num_minibatches, cfg.minibatch_size) + x.shape[1:]
                    ),
                    shuffled_traj,
                ),
                shuffled_adv.reshape((cfg.num_minibatches, cfg.minibatch_size)),
                shuffled_targets.reshape((cfg.num_minibatches, cfg.minibatch_size)),
            )
            train_state, loss_info = jax.lax.scan(
                update_minibatch, train_state, minibatches
            )
            return (train_state, flat_traj, flat_adv, flat_targets, rng), loss_info

        def update_step(runner_state):
            state = runner_state
            state, (traj, rollout_info) = jax.lax.scan(
                env_step, state, None, length=cfg.rollout_length
            )

            last_obs_in = (
                normalize(
                    state.obs, state.obs_norm, eps=cfg.norm_eps, clip=cfg.obs_clip
                )
                if cfg.normalize_obs
                else state.obs
            )
            last_value = agent.apply(
                state.train_state.params, last_obs_in, method=agent.critic
            )

            advantages, targets = compute_gae(
                traj, last_value, cfg.gamma, cfg.gae_lambda
            )
            flat_traj = jax.tree_util.tree_map(
                lambda x: x.reshape((cfg.batch_size,) + x.shape[2:]), traj
            )
            flat_adv = advantages.reshape((cfg.batch_size,))
            flat_targets = targets.reshape((cfg.batch_size,))

            (train_state, _, _, _, rng), loss_info = jax.lax.scan(
                update_epoch,
                (state.train_state, flat_traj, flat_adv, flat_targets, state.rng),
                None,
                length=cfg.update_epochs,
            )

            completed_mask = rollout_info.completed
            completed_returns = rollout_info.completed_return
            completed_lengths = rollout_info.completed_length
            num_completed = jnp.maximum(jnp.sum(completed_mask), 1.0)
            mean_return = jnp.sum(completed_returns) / num_completed
            mean_length = jnp.sum(completed_lengths) / num_completed

            losses, aux = loss_info
            actor_losses, value_losses, entropies = aux
            is_finite = jnp.array(
                [
                    jnp.all(jnp.isfinite(traj.obs)),
                    jnp.all(jnp.isfinite(traj.reward)),
                    jnp.all(jnp.isfinite(losses)),
                    jnp.all(jnp.isfinite(actor_losses)),
                    jnp.all(jnp.isfinite(value_losses)),
                ]
            ).all()

            metrics = {
                "mean_episode_return": mean_return,
                "mean_episode_length": mean_length,
                "loss": jnp.mean(losses),
                "actor_loss": jnp.mean(actor_losses),
                "value_loss": jnp.mean(value_losses),
                "entropy": jnp.mean(entropies),
                "is_finite": is_finite.astype(jnp.float32),
            }

            next_runner_state = RunnerState(
                train_state=train_state,
                obs=state.obs,
                rng=rng,
                obs_norm=state.obs_norm,
                rew_norm=state.rew_norm,
                running_returns=state.running_returns,
                episode_returns=state.episode_returns,
                episode_lengths=state.episode_lengths,
            )
            return next_runner_state, metrics

        return jax.jit(update_step)