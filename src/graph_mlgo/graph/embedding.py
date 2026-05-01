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
    def get_embedding_dim(self, node_feat_dim: int, global_feat_dim: int) -> int:
        pass
        
class TrivialEmbedder(Embedder):
    def embed(self, edge: "Edge", graph: 'Graph') -> np.ndarray:
        u, v = edge
        feat_u = graph.nodes[u].features
        feat_v = graph.nodes[v].features

        global_feat = graph.get_global_features()
        return np.concatenate([feat_u, feat_v, global_feat])

    def get_embedding_dim(self, node_feat_dim: int, global_feat_dim: int) -> int:
        return 2 * node_feat_dim + global_feat_dim