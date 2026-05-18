from dataclasses import dataclass
from typing import Literal


@dataclass
class GraphSageConfig:
    dataset_path: str

    depth: int = 3
    num_neighbours: int = 5
    hidden_dim: int = 1024
    output_dim: int = 1024
    aggregator_type: Literal["mean"] = "mean"

    seed: int = 42
    lr: float = 1e-3
    max_grad_norm: float = 1.0

    num_epochs: int = 10
    num_batches: int = 20
    batch_size: int = 128
    num_negatives: int = 10

    checkpoint_every_updates: int = 10000
