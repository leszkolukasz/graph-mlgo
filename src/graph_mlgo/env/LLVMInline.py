import random
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from datasets import Dataset
from flax import struct
from gymnasium import spaces
from loguru import logger

from graph_mlgo.graph.embedding import Embedder, EmbeddingParts
from graph_mlgo.graph.graph import Edge, Graph


@struct.dataclass
class Observation:
    embedding: jnp.ndarray
    parts: EmbeddingParts

    def to_device(self, device: jax.Device) -> "Observation":
        return Observation(
            embedding=jax.device_put(self.embedding, device),
            parts=self.parts.to_device(device),
        )

    def to_cpu(self) -> "Observation":
        return self.to_device(jax.devices("cpu")[0])

    def to_gpu(self) -> "Observation":
        gpus = jax.devices("gpu")
        if not gpus:
            raise RuntimeError("No GPU devices available for Observation")
        return self.to_device(gpus[0])


class LLVMInlineEnv(gym.Env):
    dataset: Dataset
    embedder: Embedder
    reward_density: int | None

    def __init__(
        self, dataset: Dataset, embedder: Embedder, reward_density: int | None = None
    ):
        super(LLVMInlineEnv, self).__init__()

        self.dataset = dataset
        self.embedder = embedder
        self.reward_density = reward_density

        # 0 = no inline, 1 = inline
        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.embedder.get_embedding_dim(),),
            dtype=np.float32,
        )

        self.graph: Graph | None = None
        self.baseline_size: int = 0
        self.edge_iterator = None
        self.current_edge: Edge | None = None
        self.step_count: int = 0

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict]:
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
        self.step_count = 0

        embed, parts = self.graph.get_edge_embedding(self.current_edge, self.embedder)
        return Observation(embedding=embed, parts=parts), {}

    def step(
        self, action: int | np.ndarray | jnp.ndarray
    ) -> tuple[Observation, float, bool, bool, dict]:
        assert self.graph is not None
        assert self.current_edge is not None
        assert self.edge_iterator is not None

        if isinstance(action, np.ndarray) or isinstance(action, jnp.ndarray):
            assert action.shape == (), f"Expected scalar action, got shape {action.shape}"
            action = action.item()

        if action == 1:
            self.graph.inline(self.current_edge)

        self.current_edge = next(self.edge_iterator, None)
        self.step_count += 1

        terminated = self.current_edge is None
        truncated = False

        if (
            self.reward_density is not None
            and self.step_count % self.reward_density == 0
            or terminated
        ):
            current_size = self.graph.calc_native_size()
            reward = float(self.baseline_size - current_size)
            self.baseline_size = current_size
        else:
            reward = 0.0

        info = {}

        if terminated:
            # final_ir = str(self.graph.module)

            # with open("baseline.ll", "w") as f:
            #     f.write(self.baseline_ir)
            # with open("final.ll", "w") as f:
            #     f.write(final_ir)

            info["gain"] = reward

            obs_shape = self.observation_space.shape
            assert obs_shape is not None

            obs = Observation(
                embedding=jnp.zeros(obs_shape, dtype=jnp.float32),
                parts=EmbeddingParts.empty(embed_dim=obs_shape[0]),
            )
        else:
            embed, parts = self.graph.get_edge_embedding(
                self.current_edge, self.embedder
            )
            obs = Observation(embedding=embed, parts=parts)

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
