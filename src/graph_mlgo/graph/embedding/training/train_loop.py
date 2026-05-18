import datetime
import os
import sys
import time
from typing import cast

import jax
import numpy as np
import orbax.checkpoint
from loguru import logger
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from tqdm import tqdm

from graph_mlgo.dataset import ComPileDataset
from graph_mlgo.graph import Graph
from graph_mlgo.graph.embedding import GraphSAGENet
from graph_mlgo.graph.embedding.aggregator import NAME_TO_CLASS
from graph_mlgo.graph.embedding.config import GraphSageConfig
from graph_mlgo.graph.embedding.training.trainer import (
    GraphSAGERunnerState,
    GraphSAGETrainer,
)
from graph_mlgo.graph.embedding.utils import sample_training_batches

RUNNING_STAT_WINDOW = 100
CHECKPOINT_DIR = os.path.abspath("./models/graphsage")


def run_training(config: GraphSageConfig):
    dataset = ComPileDataset(config.dataset_path)

    logger.info(
        f"Dataset loaded with {len(dataset.train)} training graphs and {len(dataset.test)} test graphs."
    )
    logger.info(
        f"Epochs: {config.num_epochs}, Batches per graph: {config.num_batches}, "
        f"Batch size (u,v pairs): {config.batch_size}, Negatives (Q): {config.num_negatives}"
    )

    model = GraphSAGENet(
        depth=config.depth,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        aggregator_cls=NAME_TO_CLASS[config.aggregator_type],
    )

    trainer = GraphSAGETrainer(model=model, config=config)
    update_fn = trainer.make_update_fn()

    options = CheckpointManagerOptions(max_to_keep=3, create=True)
    mngr = CheckpointManager(CHECKPOINT_DIR, options=options)

    rng = jax.random.PRNGKey(config.seed)
    rng, train_rng = jax.random.split(rng, 2)

    start = time.time()
    runner_state = trainer.init_runner(train_rng)
    start_update = 0

    latest_step = mngr.latest_step()
    if latest_step is not None:
        logger.info(f"Found checkpoint at update step {latest_step}. Restoring...")
        runner_state = cast(
            GraphSAGERunnerState,
            mngr.restore(
                latest_step, args=orbax.checkpoint.args.PyTreeRestore(item=runner_state)
            ),
        )
        start_update = latest_step + 1
    else:
        logger.info("No checkpoints found. Starting training from scratch.")

    total_updates = config.num_epochs * len(dataset.train) * config.num_batches

    bar = tqdm(
        range(start_update, total_updates),
        desc="Training GraphSAGE",
        unit="update",
        initial=start_update,
        total=total_updates,
    )

    last_k_losses = []
    last_k_pos_losses = []
    last_k_neg_losses = []

    update_idx = 0

    start_epoch = start_update // (len(dataset.train) * config.num_batches)

    for epoch in range(start_epoch, config.num_epochs):
        for graph_idx, llvm_sample in enumerate(dataset.train):
            graph = Graph(llvm_sample["content"])

            batches = sample_training_batches(
                graph=graph,
                num_batches=config.num_batches,
                batch_size=config.batch_size,
                num_negatives=config.num_negatives,
            )

            if not batches:
                # logger.warning(
                #     f"Graph {graph_idx} has no valid training samples. Skipping."
                # )
                continue

            for batch_data in batches:
                if update_idx < start_update:
                    update_idx += 1
                    continue

                runner_state, metrics = update_fn(runner_state, batch_data, graph)

                last_k_losses.append(float(metrics["loss"]))
                last_k_pos_losses.append(float(metrics["pos_loss"]))
                last_k_neg_losses.append(float(metrics["neg_loss"]))

                while len(last_k_losses) > RUNNING_STAT_WINDOW:
                    last_k_losses.pop(0)
                    last_k_pos_losses.pop(0)
                    last_k_neg_losses.pop(0)

                step_log = {
                    "train/loss": float(metrics["loss"]),
                    "train/pos_loss": float(metrics["pos_loss"]),
                    "train/neg_loss": float(metrics["neg_loss"]),
                    "train/running_loss": float(np.mean(last_k_losses)),
                    "epoch": epoch,
                    "graph_idx": graph_idx,
                }

                # wandb.log(step_log, step=update_idx)

                postfix_dict = {
                    "loss": f"{float(metrics['loss']):.4f}",
                    "avg_loss": f"{np.mean(last_k_losses):.4f}",
                    "epoch": f"{epoch}/{config.num_epochs}",
                }
                bar.set_postfix(postfix_dict)
                bar.update(1)

                do_checkpoint = (
                    update_idx % config.checkpoint_every_updates == 0
                    or (update_idx + 1) == total_updates
                )

                if do_checkpoint and update_idx > start_update:
                    jax.tree_util.tree_map(
                        lambda x: (
                            x.block_until_ready()
                            if hasattr(x, "block_until_ready")
                            else x
                        ),
                        runner_state,
                    )
                    mngr.save(
                        update_idx, args=orbax.checkpoint.args.PyTreeSave(runner_state)
                    )

                update_idx += 1

    mngr.wait_until_finished()
    logger.info("GraphSAGE Training completed after %.2f seconds.", time.time() - start)


if __name__ == "__main__":
    config = GraphSageConfig(
        dataset_path="./data/ComPile-1.0GB",
    )

    if len(sys.argv) > 1:
        run_id = str(sys.argv[1])
        logger.info(f"Run id: {run_id}")
    else:
        run_id = f"graphsage_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        logger.info(f"No run id provided. Using timestamp: {run_id}")

    # wandb.init(
    #     project="rl",
    #     name=run_id,
    #     id=run_id,
    #     resume="allow",
    #     config=asdict(config),
    # )

    run_training(config)
