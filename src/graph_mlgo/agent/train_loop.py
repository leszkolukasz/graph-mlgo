import time
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from loguru import logger

from graph_mlgo.agent.trainer import PPOTrainer
from graph_mlgo.agent.evaluator import PPOEvaluator
from graph_mlgo.agent.utils import make_env
from graph_mlgo.agent.networks import PPOAgent
from graph_mlgo.agent.config import PPOConfig
from graph_mlgo.dataset import ComPileDataset
from graph_mlgo.graph.embedding import TrivialEmbedder

def run_training(config: PPOConfig):
    dataset = ComPileDataset(config.dataset_path)
    embedder = TrivialEmbedder()

    logger.info(f"Dataset loaded with {len(dataset.train)} training samples and {len(dataset.test)} test samples.")
    logger.info(f"Using embedder: {embedder.__class__.__name__}")
    logger.info(f"Batch size: {config.batch_size}, Num updates: {config.num_updates}, Rollout length: {config.rollout_length}")
    logger.info(f"Num minibatches: {config.num_minibatches}, Minibatch size: {config.minibatch_size}")

    env = make_env(dataset=dataset.train, embedder=embedder, num_envs=config.num_envs, episode_length=config.episode_length)
    eval_env = make_env(dataset=dataset.test, embedder=embedder, num_envs=config.eval_num_envs, episode_length=config.episode_length)

    agent = PPOAgent(
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
    )

    trainer = PPOTrainer(config, env, agent)
    evaluator = PPOEvaluator(config, eval_env, agent)
    update_fn = trainer.make_update_fn()
    eval_fn = evaluator.make_eval_fn()

    rng = jax.random.PRNGKey(config.seed)
    rng, train_rng, eval_rng = jax.random.split(rng, 3)

    start = time.time()
    runner_state = trainer.init_runner(train_rng)

    train_metrics = []
    eval_steps = []
    eval_returns = []

    eval_every_updates = max(1, config.eval_every_env_steps // config.batch_size)
    bar = tqdm(range(config.num_updates), desc="Training PPO", unit="update")

    avg_ret_sum = 0.0
    avg_loss_sum = 0.0

    for update_idx in bar:
        runner_state, metrics = update_fn(runner_state)
        train_metrics.append(metrics)

        do_eval = (
            update_idx % eval_every_updates == 0
            or (update_idx + 1) == config.num_updates
        )

        avg_ret_sum += float(metrics["mean_episode_return"])
        avg_loss_sum += float(metrics["loss"])

        bar.set_postfix({
            "last_ret": f"{float(metrics['mean_episode_return']):.1f}",
            "last_len": f"{float(metrics['mean_episode_length']):.1f}",
            "last_loss": f"{float(metrics['loss']):.3f}",
            "avg_ret": f"{avg_ret_sum / (update_idx + 1):.1f}",
            "avg_loss": f"{avg_loss_sum / (update_idx + 1):.3f}",
        })

        if do_eval:
            eval_rng, eval_step_rng = jax.random.split(eval_rng)
            eval_metrics = eval_fn(
                runner_state.train_state, runner_state.obs_norm, eval_step_rng
            )
            eval_steps.append(update_idx * config.batch_size)
            eval_returns.append(float(jax.device_get(eval_metrics["eval_return"])))

    elapsed = time.time() - start

    metrics = jax.device_get(
        jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *train_metrics)
    )
    returns = np.asarray(metrics["mean_episode_return"])
    losses = np.asarray(metrics["loss"])
    steps = np.arange(0, config.num_updates) * config.batch_size

    result = {
        "runner_state": runner_state,
        "metrics": metrics,
    }

    return {
        "result": result,
        "metrics": metrics,
        "returns": returns,
        "losses": losses,
        "steps": steps,
        "eval_steps": np.asarray(eval_steps),
        "eval_returns": np.asarray(eval_returns),
        "elapsed": elapsed,
    }

if __name__ == "__main__":
    config = PPOConfig(
        dataset_path="./data/ComPile-1.0GB",
        num_envs=1,
        eval_num_envs=1,
    )

    result = run_training(config)

    logger.info(f"Training finished in {result['elapsed']:.2f}s")
    logger.info(f"Final train return: {float(result['returns'][-1]):.2f}")
    logger.info(f"Final eval return: {float(result['eval_returns'][-1]):.2f}")
    logger.info(f"Finite: {bool(np.all(result['metrics']['is_finite']))}")