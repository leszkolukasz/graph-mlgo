from .types import RunningNorm
import jax.numpy as jnp
import gymnasium as gym
import jax
from datasets import Dataset

from graph_mlgo.graph.embedding import Embedder
from graph_mlgo.env.LLVMInline import LLVMInlineEnv
from graph_mlgo.agent.types import Transition
from graph_mlgo.agent.config import PPOConfig

def init_running_norm(shape):
    return RunningNorm(
        mean=jnp.zeros(shape, dtype=jnp.float32),
        var=jnp.ones(shape, dtype=jnp.float32),
        count=jnp.array(1e-4, dtype=jnp.float32),
    )


def update_running_norm(norm: RunningNorm, x: jnp.ndarray) -> RunningNorm:
    batch_mean = x.mean(axis=0)
    batch_s = jnp.sum((x - batch_mean)**2, axis=0)
    old_s = norm.var * norm.count

    delta = batch_mean - norm.mean

    total_count = norm.count + x.shape[0]
    new_mean = norm.mean + delta * x.shape[0] / total_count
    new_s = old_s + batch_s + delta**2 * norm.count * x.shape[0] / total_count
    new_var = new_s / total_count

    return RunningNorm(mean=new_mean, var=new_var, count=total_count)


def normalize(
    x: jnp.ndarray,
    norm: RunningNorm,
    eps: float = 1e-6,
    clip: float | None = None,
) -> jnp.ndarray:
    y = (x - norm.mean) / (jnp.sqrt(norm.var) + eps)

    if clip is not None:
        y = jnp.clip(y, -clip, clip)

    return y

def make_env(dataset: Dataset, embedder: Embedder, num_envs: int, episode_length: int=1000):
    def make_single_env(idx: int):
        dataset_shard = dataset.shard(num_shards=num_envs, index=idx)

        env = LLVMInlineEnv(dataset=dataset_shard, embedder=embedder)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)
        
        return env

    env_fns = [
        lambda i=i: make_single_env(idx=i) 
        for i in range(num_envs)
    ]

    vec_env = gym.vector.AsyncVectorEnv(
        env_fns=env_fns,
        context="spawn"
    )

    return vec_env

def compute_gae(traj: Transition, last_value, gamma, gae_lambda):
    def scan_gae(carry, t):
        gae, next_value = carry

        delta = traj.reward[t] + gamma * next_value * (1 - traj.done[t]) - traj.value[t]
        gae = gamma * gae_lambda * gae * (1 - traj.done[t]) + delta

        return (gae, traj.value[t]), gae

    T = traj.reward.shape[0]
    time_idx = jnp.arange(T - 1, -1, -1)
    (_, _), advantages_rev = jax.lax.scan(
        scan_gae,
        (jnp.zeros_like(last_value), last_value),
        time_idx,
    )
    advantages = jnp.flip(advantages_rev, axis=0)
    targets = advantages + traj.value
    return advantages, targets

def ppo_loss(
    params,
    agent,
    batch: Transition,
    advantages: jnp.ndarray,
    targets: jnp.ndarray,
    cfg: PPOConfig,
):
    log_prob, entropy = agent.get_action_log_prob_and_entropy(
        params, batch.obs, batch.action
    )
    value = agent.apply(params, batch.obs, method=agent.critic)

    if cfg.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    log_ratio = jnp.clip(log_prob - batch.log_prob, -cfg.max_log_ratio, cfg.max_log_ratio)
    ratio = jnp.exp(log_ratio)
    unclipped = advantages * ratio
    clipped = jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages
    actor_loss = -jnp.minimum(unclipped, clipped).mean()

    value_loss = jnp.mean((value - targets)**2)

    entropy_loss = entropy.mean()

    total_loss = actor_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss
    return total_loss, (actor_loss, value_loss, entropy_loss)