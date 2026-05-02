import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import Graph, Edge

class Embedder(ABC):
    @abstractmethod
    def embed(self, edge: "Edge", graph: 'Graph') -> np.ndarray:
        global_feat = graph.get_global_features()
        node_feat = self._embed(edge, graph)
        edge_mult = np.array([graph.edges[edge]], dtype=np.float32)

        return np.concatenate([global_feat, node_feat, edge_mult])

    def get_embedding_dim(self, node_feat_dim: int, global_feat_dim: int) -> int:
        return self._get_embedding_dim(node_feat_dim) + global_feat_dim + 1

    @abstractmethod
    def _embed(self, edge: "Edge", graph: 'Graph') -> np.ndarray:
        pass

    @abstractmethod
    def _get_embedding_dim(self, node_feat_dim: int) -> int:
        pass

# dodaj krotnosc krawedzi
class TrivialEmbedder(Embedder):
    def embed(self, edge: "Edge", graph: 'Graph') -> np.ndarray:
        u, v = edge
        feat_u = graph.nodes[u].features
        feat_v = graph.nodes[v].features

        return np.concatenate([feat_u, feat_v])

    def _get_embedding_dim(self, node_feat_dim: int) -> int:
        return 2 * node_feat_dim