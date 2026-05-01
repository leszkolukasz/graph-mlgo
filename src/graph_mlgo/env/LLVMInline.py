import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Any
from loguru import logger

from datasets import Dataset
from graph_mlgo.graph.graph import Graph, Edge, Node
from graph_mlgo.graph.embedding import Embedder

class LLVMInlineEnv(gym.Env):
    def __init__(self, dataset: Dataset, embedder: Embedder):
        super(LLVMInlineEnv, self).__init__()
        
        self.dataset = dataset
        self.embedder = embedder
        
        # 0 = no inline, 1 = inline
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.embedder.get_embedding_dim(Node.get_features_dim(), Graph.get_global_embedding_dim()),),
            dtype=np.float32
        )
        
        self.graph: Graph | None = None
        self.baseline_size: int = 0
        self.edge_iterator = None
        self.current_edge: Edge | None = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        while True:
            idx = random.randint(0, len(self.dataset) - 1)
            bitcode = self.dataset[idx]["content"]
            
            self.graph = Graph(bitcode)
            self.edge_iterator = self.graph.get_inline_order()
            
            self.current_edge = next(self.edge_iterator, None)
            
            if self.current_edge is not None:
                break

            logger.warning(f"Sample idx {idx} has no edges to inline, skipping...")
                
        self.baseline_size = self.graph.calc_native_size()
        self.baseline_ir = str(self.graph.module)
        
        obs = self.graph.get_edge_embedding(self.current_edge, self.embedder)
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.graph is not None
        assert self.current_edge is not None
        assert self.edge_iterator is not None
        
        if action == 1:
            self.graph.inline(self.current_edge)
            
        self.current_edge = next(self.edge_iterator, None)
        
        terminated = (self.current_edge is None)
        truncated = False
        
        reward = 0.0
        info = {}
        
        if terminated:
            final_size = self.graph.calc_native_size()
            # final_ir = str(self.graph.module)

            # with open("baseline.ll", "w") as f:
            #     f.write(self.baseline_ir)
            # with open("final.ll", "w") as f:
            #     f.write(final_ir)
            
            reward = float(self.baseline_size - final_size)
            
            info["initial_size"] = self.baseline_size
            info["final_size"] = final_size
            info["gain"] = reward
            
            obs_shape = self.observation_space.shape
            assert obs_shape is not None

            obs = np.zeros(obs_shape, dtype=np.float32)
        else:
            obs = self.graph.get_edge_embedding(self.current_edge, self.embedder)
            
        return obs, reward, terminated, truncated, info


def sample_run():
    from graph_mlgo.dataset import ComPileDataset
    from graph_mlgo.graph.embedding import TrivialEmbedder

    dataset = ComPileDataset("./data/ComPile-1.0GB")
    embedder = TrivialEmbedder()

    print(len(dataset.train))
    print(len(dataset.test))

    env = LLVMInlineEnv(dataset=dataset.train, embedder=embedder)

    for _ in range(1000):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # action = env.action_space.sample()
            action = 1
            logger.debug(f"Action taken: {action}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        logger.info(f"Episode finished with total reward: {total_reward}")
        logger.info(f"Info: {info}")

if __name__ == "__main__":
    sample_run()
    