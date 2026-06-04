from pathlib import Path
from typing import Callable, cast

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import orbax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from loguru import logger

from graph_mlgo.agent.config import PPOConfig
from graph_mlgo.agent.networks import PPOAgent
from graph_mlgo.agent.training.types import (
    PPORunnerState,
    RolloutInfo,
    RunningNorm,
    Transition,
)
from graph_mlgo.agent.utils import (
    compute_gae,
    init_running_norm,
    normalize,
    ppo_loss,
    replace,
    update_running_norm,
)
from graph_mlgo.constants import MAX_CKPT_TO_KEEP
from graph_mlgo.env.LLVMInline import Observation
from graph_mlgo.graph.embedding import NetEmbedder
from graph_mlgo.graph.embedding.networks import EmbeddingNet
from graph_mlgo.graph.embedding.training.trainer import EmbeddingRunnerState
from graph_mlgo.graph.embedding.utils import concatenate_parts


class PPOTrainer:
    config: PPOConfig
    env: gym.Env
    agent: PPOAgent
    mngr: ocp.CheckpointManager

    def __init__(self, config: PPOConfig, env: gym.Env, agent: PPOAgent):
        self.config = config
        self.env = env
        self.agent = agent

        options = ocp.CheckpointManagerOptions(
            max_to_keep=MAX_CKPT_TO_KEEP, create=True
        )
        self.mngr = ocp.CheckpointManager(config.checkpoint_dir, options=options)

    @classmethod
    def load(
        cls,
        *,
        env: gym.Env,
        rng: jax.Array,
        checkpoint_path: str | None = None,
        config: PPOConfig | None = None,
    ) -> tuple["PPOTrainer", PPORunnerState, int]:
        assert checkpoint_path is not None or config is not None, (
            "Must provide either checkpoint_path or config to load PPOTrainer"
        )
        checkpoint_path = checkpoint_path or config.checkpoint_dir  # ty: ignore
        cp_path = Path(checkpoint_path)

        if config is None:
            config_path = cp_path / "config.yaml"
            config = PPOConfig.from_file(config_path)

        agent = PPOAgent(
            hidden_sizes=config.hidden_sizes,
        )

        trainer = cast("PPOTrainer", cls(config, env, agent))

        runner_state = trainer.init_runner(rng=rng, seed=config.seed)

        mngr = ocp.CheckpointManager(checkpoint_path)
        latest_step = mngr.latest_step()

        if latest_step is not None:
            runner_state = cast(
                PPORunnerState,
                mngr.restore(
                    latest_step, args=ocp.args.PyTreeRestore(item=runner_state)
                ),
            )
            start_update = latest_step + 1
        else:
            start_update = 0

        return trainer, runner_state, start_update

    def save_checkpoint(self, runner_state: PPORunnerState, update_idx: int) -> None:
        self.config.save()

        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            runner_state,
        )
        self.mngr.save(update_idx, args=orbax.checkpoint.args.PyTreeSave(runner_state))

    def init_runner(self, *, rng: jax.Array, seed: int) -> PPORunnerState:
        rng, init_rng = jax.random.split(rng, 2)
        obs, _ = self.env.reset(seed=seed)
        obs_dim = obs.embedding.shape[-1]

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

        return PPORunnerState(
            train_state=train_state,
            obs=obs,
            rng=rng,
            obs_norm=obs_norm,
            rew_norm=rew_norm,
            running_returns=running_returns,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
        )

    make_update_fn_ret_type = Callable[
        [PPORunnerState, EmbeddingRunnerState | None],
        tuple[PPORunnerState, EmbeddingRunnerState | None, dict[str, jnp.ndarray]],
    ]

    def make_update_fn(self) -> make_update_fn_ret_type:
        cfg = self.config
        env = self.env
        agent = self.agent

        if isinstance(env.unwrapped.embedder, NetEmbedder):  # ty: ignore
            emb_net = cast(EmbeddingNet, env.unwrapped.embedder.net)  # ty: ignore
            logger.info("Found EmbeddingNet in env.")
        else:
            emb_net = None
            logger.info("No EmbeddingNet found in env.")

        def recompute_embeddings(emb_params: VariableDict | None, raw_obs: Observation):
            if emb_params is not None:
                assert emb_net is not None and raw_obs.parts.aux is not None

                batched_apply = jax.vmap(emb_net.apply, in_axes=(None, 0, 0, 0))
                aux = raw_obs.parts.aux
                new_edge_embeds = batched_apply(
                    emb_params, aux.h, aux.indices, aux.edge_types
                )
                edge_embed = jnp.concatenate(
                    [new_edge_embeds[:, 0, :], new_edge_embeds[:, 1, :]], axis=-1
                )

                if raw_obs.parts.aux.node_feat is not None:
                    node_feat = raw_obs.parts.aux.node_feat
                    edge_embed = jnp.concatenate([edge_embed, node_feat], axis=-1)

                parts = raw_obs.parts
                replaced_parts = replace(
                    parts,
                    edge_embed=edge_embed,
                )
                return concatenate_parts(replaced_parts)

            return raw_obs.embedding

        def _jax_act(
            ppo_runner_state: PPORunnerState,
        ) -> tuple[
            jnp.ndarray, jnp.ndarray, jnp.ndarray, RunningNorm, Observation, jnp.ndarray
        ]:
            raw_obs = ppo_runner_state.obs
            obs_embed = raw_obs.embedding

            if cfg.normalize_obs:
                obs_norm = update_running_norm(ppo_runner_state.obs_norm, obs_embed)
                obs_in = normalize(
                    obs_embed, obs_norm, eps=cfg.norm_eps, clip=cfg.obs_clip
                )
            else:
                obs_norm = ppo_runner_state.obs_norm
                obs_in = obs_embed

            rng, act_rng = jax.random.split(ppo_runner_state.rng)
            action, _, log_prob = agent.act(
                ppo_runner_state.train_state.params,  # ty: ignore
                obs_in,
                act_rng,
            )
            value = cast(
                jnp.ndarray,
                agent.apply(
                    ppo_runner_state.train_state.params,  # ty: ignore
                    obs_in,
                    method=agent.critic,
                ),
            )

            return action, value, log_prob, obs_norm, raw_obs, rng

        jax_act_typ = Callable[
            [PPORunnerState],
            tuple[
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                RunningNorm,
                Observation,
                jnp.ndarray,
            ],
        ]
        jax_act = cast(jax_act_typ, jax.jit(_jax_act))

        def _jax_post_step(
            runner_state: PPORunnerState,
            obs_norm: RunningNorm,
            raw_obs: Observation,
            rng: jnp.ndarray,
            action: jnp.ndarray,
            value: jnp.ndarray,
            log_prob: jnp.ndarray,
            next_obs: Observation,
            reward: jnp.ndarray,
            done: jnp.ndarray,
        ) -> tuple[PPORunnerState, Transition, RolloutInfo]:
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
                obs=raw_obs,
                action=action,
                reward=reward_out,
                done=done,
                value=value,
                log_prob=log_prob,
            )
            rollout_info = RolloutInfo(
                completed_return=completed_return,
                completed_length=completed_length,
                completed=done,
            )
            next_carry = PPORunnerState(
                train_state=runner_state.train_state,
                obs=next_obs,
                rng=rng,
                obs_norm=obs_norm,
                rew_norm=rew_norm,
                running_returns=next_running_returns,
                episode_returns=episode_returns,
                episode_lengths=episode_lengths,
            )
            return next_carry, transition, rollout_info

        jax_post_step_typ = Callable[
            [
                PPORunnerState,
                RunningNorm,
                Observation,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                Observation,
                jnp.ndarray,
                jnp.ndarray,
            ],
            tuple[PPORunnerState, Transition, RolloutInfo],
        ]
        jax_post_step = cast(jax_post_step_typ, jax.jit(_jax_post_step))

        def update_minibatch(
            ppo_train_state: TrainState,
            emb_train_state: TrainState | None,
            minibatch: tuple[Transition, jnp.ndarray, jnp.ndarray],
            obs_norm: RunningNorm,
        ) -> tuple[TrainState, TrainState | None, tuple[jnp.ndarray, jnp.ndarray]]:
            batch, adv_batch, target_batch = minibatch

            def loss_on_params(
                ppo_params: VariableDict, emb_params: VariableDict | None
            ):
                new_embedding = recompute_embeddings(emb_params, batch.obs)

                if cfg.normalize_obs:
                    new_embedding = normalize(
                        new_embedding, obs_norm, cfg.norm_eps, cfg.obs_clip
                    )

                new_obs = replace(batch.obs, embedding=new_embedding)
                new_batch = replace(batch, obs=new_obs)
                return ppo_loss(
                    ppo_params, agent, new_batch, adv_batch, target_batch, cfg
                )

            if emb_train_state is not None:
                grad_fn = jax.value_and_grad(
                    loss_on_params, argnums=(0, 1), has_aux=True
                )
                (loss, aux), (ppo_grads, emb_grads) = grad_fn(
                    ppo_train_state.params, emb_train_state.params
                )

                new_ppo_ts = ppo_train_state.apply_gradients(grads=ppo_grads)
                new_emb_ts = emb_train_state.apply_gradients(grads=emb_grads)
                return new_ppo_ts, new_emb_ts, (loss, aux)
            else:
                grad_fn = jax.value_and_grad(loss_on_params, argnums=0, has_aux=True)
                (loss, aux), ppo_grads = grad_fn(ppo_train_state.params, None)

                new_ppo_ts = ppo_train_state.apply_gradients(grads=ppo_grads)
                return new_ppo_ts, None, (loss, aux)

        def update_epoch(
            carry: tuple[
                TrainState,
                TrainState | None,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                RunningNorm,
            ],
            _,
        ) -> tuple[
            tuple[
                TrainState,
                TrainState | None,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                jnp.ndarray,
                RunningNorm,
            ],
            tuple[jnp.ndarray, jnp.ndarray],
        ]:
            ppo_ts, emb_ts, flat_traj, flat_adv, flat_targets, rng, obs_norm = carry
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

            def scan_minibatch(carry_states, mb):
                pts, sts = carry_states
                pts_new, sts_new, loss_info = update_minibatch(pts, sts, mb, obs_norm)
                return (pts_new, sts_new), loss_info

            (ppo_ts, emb_ts), loss_info = jax.lax.scan(
                scan_minibatch, (ppo_ts, emb_ts), minibatches
            )
            return (
                ppo_ts,
                emb_ts,
                flat_traj,
                flat_adv,
                flat_targets,
                rng,
                obs_norm,
            ), loss_info

        def _jax_update(
            ppo_state: PPORunnerState,
            emb_state: EmbeddingRunnerState | None,
            traj: Transition,
            rollout_info: RolloutInfo,
        ) -> tuple[PPORunnerState, EmbeddingRunnerState | None, dict[str, jnp.ndarray]]:

            emb_params = emb_state.train_state.params if emb_state is not None else None
            last_obs_embed = recompute_embeddings(emb_params, ppo_state.obs)

            last_obs_in = (
                normalize(
                    last_obs_embed,
                    ppo_state.obs_norm,
                    eps=cfg.norm_eps,
                    clip=cfg.obs_clip,
                )
                if cfg.normalize_obs
                else last_obs_embed
            )
            last_value = agent.apply(
                ppo_state.train_state.params,  # ty: ignore
                last_obs_in,
                method=agent.critic,
            )

            advantages, targets = compute_gae(
                traj,
                last_value,  # ty: ignore
                cfg.gamma,
                cfg.gae_lambda,
            )
            flat_traj = jax.tree_util.tree_map(
                lambda x: x.reshape((cfg.batch_size,) + x.shape[2:]), traj
            )
            flat_adv = advantages.reshape((cfg.batch_size,))
            flat_targets = targets.reshape((cfg.batch_size,))

            (ppo_ts, emb_ts, _, _, _, rng, _), loss_info = jax.lax.scan(
                update_epoch,
                (
                    ppo_state.train_state,
                    emb_state.train_state if emb_state is not None else None,
                    flat_traj,
                    flat_adv,
                    flat_targets,
                    ppo_state.rng,
                    ppo_state.obs_norm,
                ),
                None,
                length=cfg.update_epochs,
            )

            old_ppo_params = ppo_state.train_state.params  # ty: ignore
            new_ppo_params = ppo_ts.params

            ppo_diff_sq = jax.tree_util.tree_map(
                lambda x, y: jnp.sum((x - y) ** 2), old_ppo_params, new_ppo_params
            )
            ppo_param_change = jnp.sqrt(
                jnp.sum(jnp.array(jax.tree_util.tree_leaves(ppo_diff_sq)))
            )

            ppo_val_sq = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), new_ppo_params)
            ppo_param_norm = jnp.sqrt(
                jnp.sum(jnp.array(jax.tree_util.tree_leaves(ppo_val_sq)))
            )

            if emb_state is not None:
                old_emb_params = emb_state.train_state.params
                new_emb_params = emb_ts.params

                diff_sq = jax.tree_util.tree_map(
                    lambda x, y: jnp.sum((x - y) ** 2), old_emb_params, new_emb_params
                )
                emb_param_change = jnp.sqrt(
                    jnp.sum(jnp.array(jax.tree_util.tree_leaves(diff_sq)))
                )

                val_sq = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), new_emb_params)
                emb_param_norm = jnp.sqrt(
                    jnp.sum(jnp.array(jax.tree_util.tree_leaves(val_sq)))
                )
            else:
                emb_param_change = jnp.array(0.0, dtype=jnp.float32)
                emb_param_norm = jnp.array(0.0, dtype=jnp.float32)

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
                    jnp.all(jnp.isfinite(traj.obs.embedding)),
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
                "emb_param_change": emb_param_change,
                "emb_param_norm": emb_param_norm,
                "ppo_param_change": ppo_param_change,
                "ppo_param_norm": ppo_param_norm,
            }

            next_runner_state = PPORunnerState(
                train_state=ppo_ts,
                obs=ppo_state.obs,
                rng=rng,
                obs_norm=ppo_state.obs_norm,
                rew_norm=ppo_state.rew_norm,
                running_returns=ppo_state.running_returns,
                episode_returns=ppo_state.episode_returns,
                episode_lengths=ppo_state.episode_lengths,
            )

            next_emb_state = None
            if emb_state is not None:
                next_emb_state = EmbeddingRunnerState(
                    train_state=emb_ts,
                )

            return next_runner_state, next_emb_state, metrics

        jax_update_typ = Callable[
            [PPORunnerState, EmbeddingRunnerState | None, Transition, RolloutInfo],
            tuple[PPORunnerState, EmbeddingRunnerState | None, dict[str, jnp.ndarray]],
        ]
        jax_update = cast(jax_update_typ, jax.jit(_jax_update))

        def update_step(
            runner_state: PPORunnerState,
            emb_runner_state: EmbeddingRunnerState | None = None,
        ) -> tuple[PPORunnerState, EmbeddingRunnerState | None, dict[str, jnp.ndarray]]:
            state = runner_state
            transitions: list[Transition] = []
            rollout_infos: list[RolloutInfo] = []

            if emb_runner_state is not None:
                embedder = env.unwrapped.embedder  # type: ignore
                assert isinstance(embedder, NetEmbedder)
                embedder.params = emb_runner_state.train_state.params
                # logger.info("Updated embedder parameters for PPO update.")

            for _ in range(cfg.rollout_length):
                action, value, log_prob, obs_norm, raw_obs, rng = jax_act(state)

                next_obs, reward, terminated, truncated, _ = env.step(action)

                next_obs_jax = next_obs.to_gpu()
                reward_jax = jnp.asarray(reward, dtype=jnp.float32)
                done_jax = jnp.logical_or(terminated, truncated).astype(jnp.float32)

                state, transition, rollout_info = jax_post_step(
                    state,
                    obs_norm,
                    raw_obs,
                    rng,
                    action,
                    value,
                    log_prob,
                    next_obs_jax,
                    reward_jax,
                    done_jax,
                )

                transitions.append(transition)
                rollout_infos.append(rollout_info)

            traj = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *transitions)
            stacked_rollout_info = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs), *rollout_infos
            )

            next_runner_state, next_emb_runner_state, metrics = jax_update(
                state, emb_runner_state, traj, stacked_rollout_info
            )

            return next_runner_state, next_emb_runner_state, metrics

        return update_step
