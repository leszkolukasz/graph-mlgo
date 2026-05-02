import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import Graph, Edge

class Embedder(ABC):
    def embed(self, edge: "Edge", graph: 'Graph') -> np.ndarray:
        global_feat = graph.get_global_features()
        node_feat = self._embed(edge, graph)
        edge_mult = np.array([graph.edges[edge]], dtype=np.float32)

        caller, callee = edge
        total_args = 0
        const_args = 0
        
        caller_func = graph.module.get_function(caller)
        for block in caller_func.blocks:
            for instr in block.instructions:
                if instr.opcode in ("call", "invoke"):
                    ops = list(instr.operands)
                    if ops and getattr(ops[-1], "name", "") == callee:
                        args = ops[:-1]
                        total_args += len(args)
                        const_args += sum(1 for arg in args if getattr(arg, "is_constant", False))
            
        const_ratio = np.array([float(const_args) / total_args if total_args > 0 else 0.0], dtype=np.float32)

        return np.concatenate([global_feat, node_feat, edge_mult, const_ratio])

    def get_embedding_dim(self, node_feat_dim: int, global_feat_dim: int) -> int:
        return self._get_embedding_dim(node_feat_dim) + global_feat_dim + 2

    @abstractmethod
    def _embed(self, edge: "Edge", graph: 'Graph') -> np.ndarray:
        pass

    @abstractmethod
    def _get_embedding_dim(self, node_feat_dim: int) -> int:
        pass

class TrivialEmbedder(Embedder):
    def _embed(self, edge: "Edge", graph: 'Graph') -> np.ndarray:
        u, v = edge
        feat_u = graph.nodes[u].features
        feat_v = graph.nodes[v].features

        return np.concatenate([feat_u, feat_v])

    def _get_embedding_dim(self, node_feat_dim: int) -> int:
        return 2 * node_feat_dim