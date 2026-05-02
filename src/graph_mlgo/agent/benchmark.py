import os
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from loguru import logger
import orbax.checkpoint
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions

from graph_mlgo.agent.utils import normalize, make_env
from graph_mlgo.agent.networks import PPOAgent
from graph_mlgo.agent.config import PPOConfig
from graph_mlgo.agent.trainer import PPOTrainer
from graph_mlgo.dataset import ComPileDataset
from graph_mlgo.graph.graph import Graph
from graph_mlgo.graph.embedding import TrivialEmbedder
from graph_mlgo.ir import compile_module

def run_benchmark(config: PPOConfig, checkpoint_dir: str):
    dataset = ComPileDataset(config.dataset_path)
    embedder = TrivialEmbedder()

    dummy_env = make_env(dataset=dataset.test, embedder=embedder, num_envs=1, episode_length=config.episode_length)
    agent = PPOAgent(
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
    )
    trainer = PPOTrainer(config, dummy_env, agent)

    rng = jax.random.PRNGKey(config.seed)
    runner_state = trainer.init_runner(rng)

    options = CheckpointManagerOptions(max_to_keep=3, create=False)
    mngr = CheckpointManager(os.path.abspath(checkpoint_dir), options=options)
    
    latest_step = mngr.latest_step()
    if latest_step is None:
        logger.error("No checkpoint found to benchmark.")
        return

    logger.info(f"Restoring checkpoint from step {latest_step} for benchmarking...")
    runner_state = mngr.restore(
        latest_step,
        args=orbax.checkpoint.args.PyTreeRestore(item=runner_state)
    )
    
    params = runner_state.train_state.params # ty: ignore
    obs_norm = runner_state.obs_norm # ty: ignore

    @jax.jit
    def get_action(obs_in, key):
        action, _, _ = agent.act(params, obs_in, key)
        return action

    llvm_sizes = []
    agent_sizes = []

    bar = tqdm(dataset.test, desc="Benchmarking LLVM vs Agent")

    for item in bar:
        bitcode = item["content"]
        
        graph_llvm = Graph(bitcode)
        size_baseline, _ = compile_module(str(graph_llvm.module), enable_inlining=False)
        size_llvm, _ = compile_module(str(graph_llvm.module), enable_inlining=True)
        
        graph_agent = Graph(bitcode)
        for edge in graph_agent.get_inline_order():
            obs = graph_agent.get_edge_embedding(edge, embedder)
            obs_batched = jnp.expand_dims(jnp.asarray(obs, dtype=jnp.float32), axis=0)
            
            if config.normalize_obs:
                obs_in = normalize(obs_batched, obs_norm, eps=config.norm_eps, clip=config.obs_clip)
            else:
                obs_in = obs_batched
                
            rng, act_rng = jax.random.split(rng)
            action_jax = get_action(obs_in, act_rng)
            action = np.asarray(action_jax).argmax().item()
            
            if action == 1:
                graph_agent.inline(edge)
        
        size_agent, _ = compile_module(str(graph_agent.module), enable_inlining=False)
        
        llvm_sizes.append(size_baseline-size_llvm)
        agent_sizes.append(size_baseline-size_agent)

        bar.set_postfix({
            "Avg LLVM": f"{np.mean(llvm_sizes):.2f} bytes",
            "Avg Agent": f"{np.mean(agent_sizes):.2f} bytes",
            "Avg Diff": f"{np.mean(agent_sizes) - np.mean(llvm_sizes):.2f} bytes"
        })

    total_llvm = sum(llvm_sizes)
    total_agent = sum(agent_sizes)

    logger.info(f"Total size LLVM default inlining: {total_llvm} bytes")
    logger.info(f"Total size Agent inlining: {total_agent} bytes")

    logger.info(f"Average size LLVM default inlining: {total_llvm / len(llvm_sizes):.2f} bytes")
    logger.info(f"Average size Agent inlining: {total_agent / len(agent_sizes):.2f} bytes")

if __name__ == "__main__":
    config = PPOConfig(
        dataset_path="./data/ComPile-1.0GB",
        hidden_sizes=(128, 128),
    )

    checkpoint_dir = "./models"
    run_benchmark(config, checkpoint_dir)