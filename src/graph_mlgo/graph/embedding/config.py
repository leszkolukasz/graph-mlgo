from dataclasses import dataclass

from graph_mlgo.graph.embedding.aggregator import Aggregator, MeanAggregator


@dataclass
class GraphSageConfig:
    dataset_path: str

    depth: int = 2
    num_neighbours: int = 5
    hidden_dim: int = 64
    output_dim: int = 64
    aggregator_cls: type[Aggregator] = MeanAggregator

    seed: int = 42
    lr: float = 1e-4
    max_grad_norm: float = 0.5

    num_epochs: int = 10
    num_batches: int = 20
    batch_size: int = 128
    num_negatives: int = 20

    checkpoint_every_updates: int = 500
