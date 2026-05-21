import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

CHECKPOINT_DIR = os.path.abspath("./models/embedding")


@dataclass
class EmbeddingConfig:
    dataset_path: str
    checkpoint_dir: str = CHECKPOINT_DIR
    embedding_type: Literal["graphsage", "gat"] = "graphsage"

    depth: int = 4
    num_neighbours: int = 5
    hidden_dim: int = 128
    output_dim: int = 32
    aggregator_type: Literal["mean", "pool"] = "pool"

    seed: int = 42
    lr: float = 3e-4
    max_grad_norm: float = 1.0

    num_epochs: int = 10
    num_batches: int = 20
    batch_size: int = 256
    num_negatives: int = 10

    checkpoint_every_updates: int = 10000

    @classmethod
    def load(cls, path: str | Path | None = None) -> "EmbeddingConfig":
        return cls.from_file(path)

    def save(self, path: str | None = None) -> None:
        self.to_file(path or os.path.join(self.checkpoint_dir, "config.yaml"))

    @classmethod
    def from_file(cls, path: str | Path | None) -> "EmbeddingConfig":
        if path is None:
            path = os.path.join(CHECKPOINT_DIR, "config.yaml")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_file(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)
