import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from graph_mlgo.constants import MAX_EDGES
from graph_mlgo.graph.embedding.config import EmbeddingConfig

CHECKPOINT_DIR = os.path.abspath("./models/ppo")


@dataclass
class PPOConfig:
    dataset_path: str
    checkpoint_dir: str = CHECKPOINT_DIR

    embedder_path_and_type: tuple[str, Literal["graphsage", "gat"]] | None = (
        None  # load existing embedder
    )
    embedder_train_config: EmbeddingConfig | None = None  # train new embedder

    total_timesteps: int = 10_000_000
    num_envs: int = 1

    rollout_length: int = 1024
    update_epochs: int = 4
    minibatch_size: int = 512

    lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    reward_density: int | None = 5

    episode_length: int = MAX_EDGES
    hidden_sizes: tuple[int, ...] = (256, 256)
    # activation: Callable = nn.tanh
    anneal_lr: bool = True

    normalize_advantage: bool = True
    normalize_obs: bool = True
    normalize_reward: bool = True

    norm_eps: float = 1e-6
    obs_clip: float = 10.0
    reward_clip: float = 10.0
    max_log_ratio: float = 20.0

    checkpoint_every_updates: int = 50

    eval_every_updates: int = 50
    eval_num_envs: int = 1
    eval_horizon: int = 5000

    seed: int = 42

    def __post_init__(self):
        self.batch_size = self.num_envs * self.rollout_length
        self.num_minibatches = self.batch_size // self.minibatch_size
        self.num_updates = self.total_timesteps // self.batch_size
        self.checkpoint_dir = os.path.abspath(self.checkpoint_dir)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PPOConfig":
        return cls.from_file(path)

    def save(self, path: str | None = None) -> None:
        self.to_file(path or os.path.join(self.checkpoint_dir, "config.yaml"))

    @classmethod
    def from_file(cls, path: str | Path | None) -> "PPOConfig":
        if path is None:
            path = os.path.join(CHECKPOINT_DIR, "config.yaml")

        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.UnsafeLoader)

        valid_keys = {field.name for field in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        return cls(**filtered_data)

    def to_file(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)
