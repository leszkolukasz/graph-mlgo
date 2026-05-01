import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import Graph, Edge

class Embedder(ABC):
    @abstractmethod
    def embed(self, edge: "Edge", graph: 'Graph') -> np.ndarray:
        pass

    @abstractmethod
    def get_embedding_dim(self, graph: 'Graph') -> int:
        pass
        
class TrivialEmbedder(Embedder):
    def embed(self, edge: "Edge", graph: 'Graph') -> np.ndarray:
        u, v = edge
        feat_u = graph.nodes[u].features
        feat_v = graph.nodes[v].features

        global_feat = graph.get_global_features()
        return np.concatenate([feat_u, feat_v, global_feat])

    def get_embedding_dim(self, graph: 'Graph') -> int:
        sample_node = next(iter(graph.nodes.values()))
        return 2 * len(sample_node.features) + graph.get_global_embedding_dim()