from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class GraphSageConfig:
    dataset_path: str

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
    def from_file(cls, path: str | Path) -> "GraphSageConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_file(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)
