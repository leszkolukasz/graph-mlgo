import numpy as np


class Node:
    features: np.ndarray

    def __init__(self, name: str):
        self.name: str = name
        self.neighbours: set[str] = set()
        self.features: np.ndarray = np.zeros(Node.get_features_dim(), dtype=np.float32)

    @staticmethod
    def get_features_dim() -> int:
        return 10
