from typing import Callable
from dataclasses import dataclass
import flax.linen as nn

from graph_mlgo.dataset.prepare import MAX_EDGES

@dataclass
class PPOConfig:
    dataset_path: str

    total_timesteps: int = 10_000_000
    num_envs: int = 4

    rollout_length: int = 128
    update_epochs: int = 16
    minibatch_size: int = 16

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5

    episode_length: int = MAX_EDGES
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: Callable = nn.tanh
    anneal_lr: bool = True

    normalize_advantage: bool = True
    normalize_obs: bool = True
    normalize_reward: bool = True

    norm_eps: float = 1e-6
    obs_clip: float = 10.0
    reward_clip: float = 10.0
    max_log_ratio: float = 20.0

    eval_every_env_steps: int = 10_000
    eval_num_envs: int = 4
    eval_horizon: int = 1000

    seed: int = 0

    def __post_init__(self):
        self.batch_size = self.num_envs * self.rollout_length
        self.num_minibatches = (self.batch_size + self.minibatch_size - 1) // self.minibatch_size
        self.num_updates = self.total_timesteps // self.batch_size