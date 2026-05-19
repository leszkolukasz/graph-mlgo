import numpy as np

from graph_mlgo.graph.embedding.constants import NODE_FEATURES_DIM


class Node:
    features: np.ndarray

    def __init__(self, name: str):
        self.name: str = name
        self.neighbours: set[str] = set()
        self.features: np.ndarray = np.zeros(NODE_FEATURES_DIM, dtype=np.float32)
