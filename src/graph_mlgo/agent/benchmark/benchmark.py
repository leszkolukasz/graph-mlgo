import os

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from loguru import logger
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from tqdm import tqdm

from graph_mlgo.agent.config import PPOConfig
from graph_mlgo.agent.networks import PPOAgent
from graph_mlgo.agent.training.ppo.trainer import PPOTrainer
from graph_mlgo.agent.utils import load_embedder, make_env, normalize
from graph_mlgo.constants import MAX_CKPT_TO_KEEP
from graph_mlgo.dataset import ComPileDataset
from graph_mlgo.graph.graph import Graph
from graph_mlgo.ir import compile_module, compile_module_no_opt


def run_benchmark(checkpoint_dir: str):
    config = PPOConfig.from_file(os.path.join(checkpoint_dir, "config.yaml"))
    logger.info(f"Loaded config from checkpoint: {config}")

    dataset = ComPileDataset(config.dataset_path)
    rng = jax.random.PRNGKey(config.seed)
    emb_rng, train_rng, rng = jax.random.split(rng, 3)

    embedder = load_embedder(config, emb_rng)

    logger.info(
        f"Dataset loaded with {len(dataset.train)} training samples and {len(dataset.test)} test samples."
    )
    logger.info(f"Using embedder: {embedder.__class__.__name__}")

    dummy_env = make_env(
        dataset=dataset.test,
        embedder=embedder,
        num_envs=1,
        episode_length=config.episode_length,
    )

    agent = PPOAgent(
        hidden_sizes=config.hidden_sizes,
    )
    trainer = PPOTrainer(config, dummy_env, agent)

    runner_state = trainer.init_runner(rng=train_rng, seed=config.seed)

    options = CheckpointManagerOptions(max_to_keep=MAX_CKPT_TO_KEEP, create=False)
    mngr = CheckpointManager(os.path.abspath(checkpoint_dir), options=options)

    latest_step = mngr.latest_step()
    if latest_step is None:
        logger.error("No checkpoint found to benchmark.")
        return

    logger.info(f"Restoring checkpoint from step {latest_step} for benchmarking...")
    runner_state = mngr.restore(
        latest_step, args=orbax.checkpoint.args.PyTreeRestore(item=runner_state)
    )

    params = runner_state.train_state.params  # ty: ignore
    obs_norm = runner_state.obs_norm  # ty: ignore

    @jax.jit
    def get_action(obs_in, key):
        action, determ_action, _ = agent.act(params, obs_in, key)
        return action, determ_action

    llvm_gains_opt = []
    llvm_gains_no_opt = []
    agent_gains_opt = []
    agent_gains_no_opt = []
    raw_diffs = []

    llvm_perc_gains = []
    agent_perc_gains = []

    # plt.ion()
    # fig, ax = plt.subplots(figsize=(10, 6))
    # plt.show(block=False)

    bar = tqdm(dataset.test, desc="Benchmarking LLVM vs Agent")

    for idx, item in enumerate(bar):
        bitcode = item["content"]

        graph_llvm = Graph(bitcode)

        size_baseline_no_opt, _ = compile_module_no_opt(str(graph_llvm.module))
        size_baseline_opt, _ = compile_module(
            str(graph_llvm.module), enable_inlining=False
        )
        size_llvm, _ = compile_module(str(graph_llvm.module), enable_inlining=True)

        graph_agent = Graph(bitcode)
        for edge in graph_agent.get_inline_order():
            obs, _ = graph_agent.get_edge_embedding(edge, embedder)
            obs_batched = jnp.expand_dims(jnp.asarray(obs, dtype=jnp.float32), axis=0)

            if config.normalize_obs:
                obs_in = normalize(
                    obs_batched, obs_norm, eps=config.norm_eps, clip=config.obs_clip
                )
            else:
                obs_in = obs_batched

            rng, act_rng = jax.random.split(rng)
            _, determ_action_jax = get_action(obs_in, act_rng)
            determ_action = determ_action_jax.item()

            if determ_action == 1:
                graph_agent.inline(edge)

        size_agent, _ = compile_module(str(graph_agent.module), enable_inlining=False)

        # logger.info(f"Sample {idx+1}: Baseline Size = {size_baseline} bytes, LLVM Size = {size_llvm} bytes, Agent Size = {size_agent} bytes")

        llvm_gain_opt = size_baseline_opt - size_llvm
        agent_gain_opt = size_baseline_opt - size_agent

        llvm_gain_no_opt = size_baseline_no_opt - size_llvm
        agent_gain_no_opt = size_baseline_no_opt - size_agent

        llvm_perc_gain = (
            (llvm_gain_no_opt / size_baseline_no_opt) * 100
            if size_baseline_no_opt > 0
            else 0
        )
        agent_perc_gain = (
            (agent_gain_no_opt / size_baseline_no_opt) * 100
            if size_baseline_no_opt > 0
            else 0
        )

        llvm_gains_opt.append(llvm_gain_opt)
        agent_gains_opt.append(agent_gain_opt)
        llvm_gains_no_opt.append(llvm_gain_no_opt)
        agent_gains_no_opt.append(agent_gain_no_opt)
        llvm_perc_gains.append(llvm_perc_gain)
        agent_perc_gains.append(agent_perc_gain)

        raw_diffs.append(agent_gain_opt - llvm_gain_opt)

        avg_llvm_gain_opt = np.mean(llvm_gains_opt)
        avg_agent_gain_opt = np.mean(agent_gains_opt)
        avg_llvm_gain_no_opt = np.mean(llvm_gains_no_opt)
        avg_agent_gain_no_opt = np.mean(agent_gains_no_opt)
        avg_llvm_perc_gain = np.mean(llvm_perc_gains)
        avg_agent_perc_gain = np.mean(agent_perc_gains)

        bar.set_postfix(
            {
                "LLVM (Opt)": f"{avg_llvm_gain_opt:.2f} bytes",
                "Agent (Opt)": f"{avg_agent_gain_opt:.2f} bytes",
                "LLVM (No Opt)": f"{avg_llvm_gain_no_opt:.2f} bytes",
                "Agent (No Opt)": f"{avg_agent_gain_no_opt:.2f} bytes",
                "Diff (No Opt)": f"{(avg_agent_gain_no_opt - avg_llvm_gain_no_opt):.2f} bytes",
                "LLVM % Gain": f"{avg_llvm_perc_gain:.2f}%",
                "Agent % Gain": f"{avg_agent_perc_gain:.2f}%",
            }
        )

        # if idx % 20 == 0 and idx > 0:
        #     ax.clear()
        #     ax.hist(raw_diffs, bins=100, alpha=0.7, color="blue")

        #     ax.set_title(f"Distribution of Agent vs LLVM Size Gains (n={idx + 1})")
        #     ax.set_xlabel("Agent Gain - LLVM Gain (bytes)")
        #     ax.set_ylabel("Frequency")
        #     ax.grid(True)

        #     fig.canvas.draw()
        #     fig.canvas.flush_events()

    logger.info(f"Avg LLVM Gain (Opt): {avg_llvm_gain_opt:.2f} bytes")
    logger.info(f"Avg Agent Gain (Opt): {avg_agent_gain_opt:.2f} bytes")
    logger.info(f"Avg LLVM Gain (No Opt): {avg_llvm_gain_no_opt:.2f} bytes")
    logger.info(f"Avg Agent Gain (No Opt): {avg_agent_gain_no_opt:.2f} bytes")
    logger.info(f"Avg LLVM % Gain: {avg_llvm_perc_gain:.2f}%")
    logger.info(f"Avg Agent % Gain: {avg_agent_perc_gain:.2f}%")

    return (
        llvm_gains_opt,
        agent_gains_opt,
        llvm_gains_no_opt,
        agent_gains_no_opt,
        llvm_perc_gains,
        agent_perc_gains,
    )


if __name__ == "__main__":
    # checkpoint_dir = "./models_final/ppo/paper_ppo_sage_contr"
    checkpoint_dirs = [
        # "./models_final/ppo/paper_ppo_256_best",
        # "./models_final/ppo/paper_ppo_gat_pretrain_best",
        # "./models_final/ppo/paper_ppo_gat_pretrain_noretain",
        # "./models_final/ppo/paper_ppo_gat_pretrain_noretain_best",
        # "./models_final/ppo/paper_ppo_gat_sparse_3",
        # "./models_final/ppo/paper_ppo_gat_sparse_3_best",
        # "./models_final/ppo/paper_ppo_sage_3_best",
        # "./models_final/ppo/paper_ppo_sage_contr_best",
        # "./models_final/ppo/paper_ppo_sparse_3_best",
        # "./models_final/ppo/paper_ppo_sage_pretrain_best",
        # "./models_final/ppo/paper_ppo_sage_sparse_4",
        # "./models_final/ppo/paper_ppo_sage_pretrain_noretain",
        # "./models_final/ppo/paper_ppo_sage_pretrain_noretain_best",
        "./models_final/ppo/paper_ppo_sparse_3_mlgo",
    ]

    for checkpoint_dir in checkpoint_dirs:
        (
            avg_llvm_gains_opt,
            avg_agent_gains_opt,
            avg_llvm_gains_no_opt,
            avg_agent_gains_no_opt,
            avg_llvm_perc_gains,
            avg_agent_perc_gains,
        ) = run_benchmark(checkpoint_dir)

        np.save(
            os.path.join(checkpoint_dir, "llvm_gains_opt.npy"),
            np.array(avg_llvm_gains_opt),
        )
        np.save(
            os.path.join(checkpoint_dir, "agent_gains_opt.npy"),
            np.array(avg_agent_gains_opt),
        )
        np.save(
            os.path.join(checkpoint_dir, "llvm_gains_no_opt.npy"),
            np.array(avg_llvm_gains_no_opt),
        )
        np.save(
            os.path.join(checkpoint_dir, "agent_gains_no_opt.npy"),
            np.array(avg_agent_gains_no_opt),
        )
        np.save(
            os.path.join(checkpoint_dir, "llvm_perc_gains.npy"),
            np.array(avg_llvm_perc_gains),
        )
        np.save(
            os.path.join(checkpoint_dir, "agent_perc_gains.npy"),
            np.array(avg_agent_perc_gains),
        )
