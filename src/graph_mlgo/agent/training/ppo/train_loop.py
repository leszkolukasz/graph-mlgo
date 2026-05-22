import datetime
import sys
import time
from dataclasses import asdict

import jax
import numpy as np
from loguru import logger
from tqdm import tqdm

import wandb
from graph_mlgo.agent.config import PPOConfig
from graph_mlgo.agent.training.ppo.evaluator import PPOEvaluator
from graph_mlgo.agent.training.ppo.trainer import PPOTrainer
from graph_mlgo.agent.utils import load_embedder, make_env
from graph_mlgo.dataset import ComPileDataset
from graph_mlgo.graph.embedding import NetEmbedder
from graph_mlgo.graph.embedding.config import EmbeddingConfig
from graph_mlgo.graph.embedding.training.trainer import (
    EmbeddingRunnerState,
    EmbeddingTrainer,
)

RUNNING_STAT_WINDOW = 100
ENABLE_WANDB = False


def run_training(config: PPOConfig | None):
    if config is None:
        config = PPOConfig.load()
        logger.info(f"Loaded PPO config from checkpoint: {config}")
    else:
        logger.info(f"Using provided PPO config: {config}")

    dataset = ComPileDataset(config.dataset_path)

    rng: jax.Array = jax.random.PRNGKey(config.seed)
    ppo_rng, emb_trainer_rng, emb_rng, eval_rng = jax.random.split(rng, 4)

    embedding_trainer: EmbeddingTrainer | None = None
    embedding_runner_state: EmbeddingRunnerState | None = None

    if config.embedder_train_config is not None:
        embedding_trainer, embedding_runner_state, _ = EmbeddingTrainer.load(
            rng=emb_trainer_rng, config=config.embedder_train_config
        )

    embedder = load_embedder(config, emb_rng)

    logger.info(
        f"Dataset loaded with {len(dataset.train)} training samples and {len(dataset.test)} test samples."
    )
    logger.info(f"Using embedder: {embedder.__class__.__name__}")
    logger.info(
        f"Batch size: {config.batch_size}, Num updates: {config.num_updates}, Rollout length: {config.rollout_length}"
    )
    logger.info(
        f"Num minibatches: {config.num_minibatches}, Minibatch size: {config.minibatch_size}"
    )

    env = make_env(
        dataset=dataset.train,
        embedder=embedder,
        num_envs=config.num_envs,
        episode_length=config.episode_length,
    )
    eval_env = make_env(
        dataset=dataset.test,
        embedder=embedder,
        num_envs=config.eval_num_envs,
        episode_length=config.episode_length,
    )

    trainer, ppo_runner_state, start_update = PPOTrainer.load(
        env=env, config=config, rng=ppo_rng
    )
    evaluator = PPOEvaluator(config, eval_env, trainer.agent)
    update_fn = trainer.make_update_fn()
    eval_fn = evaluator.make_eval_fn()

    start = time.time()
    if start_update > 0:
        logger.info(f"Found PPO checkpoint. Resuming from update {start_update}.")
    else:
        logger.info(
            "No PPO checkpoints found in the folder. Starting training from scratch."
        )

    bar = tqdm(
        range(start_update, config.num_updates),
        desc="Training PPO",
        unit="update",
        initial=start_update,
        total=config.num_updates,
    )

    last_k_returns = []
    last_k_losses = []
    avg_length_sum = 0.0
    last_eval_return = None

    for update_idx in bar:
        ppo_runner_state, embedding_runner_state, metrics = update_fn(
            ppo_runner_state, embedding_runner_state
        )

        do_eval = (
            update_idx % config.eval_every_updates == 0
            or (update_idx + 1) == config.num_updates
        )

        do_checkpoint = (
            update_idx % config.checkpoint_every_updates == 0
            or (update_idx + 1) == config.num_updates
        )

        last_k_returns.append(float(metrics["mean_episode_return"]))
        last_k_losses.append(float(metrics["loss"]))
        avg_length_sum += float(metrics["mean_episode_length"])

        while len(last_k_returns) > RUNNING_STAT_WINDOW:
            last_k_returns.pop(0)
            last_k_losses.pop(0)

        step_log = {
            "train/mean_return": float(metrics["mean_episode_return"]),
            "train/mean_length": float(
                avg_length_sum / (update_idx - start_update + 1)
            ),
            "train/running_mean_return": float(np.mean(last_k_returns)),
            "train/running_loss": float(np.mean(last_k_losses)),
            "train/loss": float(metrics["loss"]),
            "train/actor_loss": float(metrics["actor_loss"]),
            "train/value_loss": float(metrics["value_loss"]),
            "train/entropy": float(metrics["entropy"]),
            "train/is_finite": float(metrics["is_finite"]),
            "env_step": update_idx * config.batch_size,
        }

        postfix_dict = {
            "ret": f"{float(metrics['mean_episode_return']):.1f}",
            "loss": f"{float(metrics['loss']):.3f}",
            "avg_ret": f"{np.mean(last_k_returns):.1f}",
            "avg_loss": f"{np.mean(last_k_losses):.3f}",
        }

        if do_eval:
            eval_rng, eval_step_rng = jax.random.split(eval_rng)

            if embedding_runner_state is not None:
                # logger.info("Updating embedder parameters for evaluation.")
                embedder = env.unwrapped.embedder  # ty: ignore
                assert isinstance(embedder, NetEmbedder)
                embedder.params = embedding_runner_state.train_state.params

            eval_metrics = eval_fn(
                ppo_runner_state.train_state, ppo_runner_state.obs_norm, eval_step_rng
            )

            current_eval_return = float(jax.device_get(eval_metrics["eval_return"]))
            last_eval_return = current_eval_return
            step_log["eval/return"] = current_eval_return

        if ENABLE_WANDB:
            wandb.log(step_log, step=update_idx)

        if do_checkpoint:
            trainer.save_checkpoint(ppo_runner_state, update_idx)
            if embedding_trainer is not None:
                assert embedding_runner_state is not None
                embedding_trainer.save_checkpoint(embedding_runner_state, update_idx)

        if last_eval_return is not None:
            postfix_dict["eval_ret"] = f"{last_eval_return:.1f}"

        bar.set_postfix(postfix_dict)

    trainer.mngr.wait_until_finished()
    if embedding_trainer is not None:
        embedding_trainer.mngr.wait_until_finished()

    logger.info("Training completed after %.2f seconds.", time.time() - start)


if __name__ == "__main__":
    typ = "graphsage"

    embedding_config = EmbeddingConfig(
        dataset_path="./data/ComPile-1.0GB",
        embedding_type=typ,
        checkpoint_dir=f"./models/embedding/{typ}",
    )

    config = PPOConfig(
        dataset_path="./data/ComPile-1.0GB",
        # embedder_path="./models/graphsage",
        embedder_train_config=embedding_config,
        hidden_sizes=(128, 128),
    )

    if len(sys.argv) > 1:
        run_id = str(sys.argv[1])
        logger.info(f"Run id: {run_id}")
        assert ENABLE_WANDB, "WandB must be enabled to use run ids."
    else:
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f"No run id provided. Using timestamp: {run_id}")

    if ENABLE_WANDB:
        wandb.init(
            project="rl", name=run_id, id=run_id, resume="allow", config=asdict(config)
        )

    result = run_training(config)

    logger.info(f"Training finished in {result['elapsed']:.2f}s")
    logger.info(f"Final train return: {float(result['returns'][-1]):.2f}")
    logger.info(f"Final eval return: {float(result['eval_returns'][-1]):.2f}")
    logger.info(f"Finite: {bool(np.all(result['metrics']['is_finite']))}")
