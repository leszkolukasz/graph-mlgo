import os
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

NBINS = 1000


def plot_benchmark_for_metric(
    checkpoint_dirs: list[tuple[str, str]],
    *,
    value_type: str,
    plot_name: str,
    yaxis_name: str,
    diff: Literal["llvm_minus_agent", "agent_minus_llvm"] | None = None,
    y_limit: tuple[float, float] | None = None,
    bin_width: float | None = None,
):
    logger.info(f"Plotting {plot_name}...")

    fig = go.Figure()

    first_dir = checkpoint_dirs[0][0]
    llvm_path = os.path.join(first_dir, f"llvm_{value_type}.npy")

    llvm_values = np.load(llvm_path)

    num_rows = len(checkpoint_dirs) if diff is not None else len(checkpoint_dirs) + 1

    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.02,
    )

    min_5_quartile = float("inf")
    max_95_quartile = float("-inf")
    current_row = 1

    hist_kwargs = (
        dict(xbins=dict(size=bin_width))
        if bin_width is not None
        else dict(nbinsx=NBINS)
    )

    if diff is None:
        llvm_mean = np.mean(llvm_values)
        llvm_median = np.median(llvm_values)

        for i in range(1, len(checkpoint_dirs)):
            llvm_path_other = os.path.join(
                checkpoint_dirs[i][0], f"llvm_{value_type}.npy"
            )
            llvm_values_other = np.load(llvm_path_other)

            assert np.array_equal(llvm_values, llvm_values_other)

        min_5_quartile = min(min_5_quartile, np.percentile(llvm_values, 5))
        max_95_quartile = max(max_95_quartile, np.percentile(llvm_values, 95))
        q1, q2, q3 = np.percentile(llvm_values, [25, 50, 75])
        success_rate = np.mean(llvm_values >= 0) * 100

        logger.info(
            f"Baseline LLVM (Oz), Mean: {llvm_mean:.2f}, Median: {llvm_median:.2f}, Q1: {q1:.2f}, Q3: {q3:.2f}, Success Rate: {success_rate:.2f}%"
        )

        fig.add_trace(
            go.Histogram(
                x=llvm_values,
                name="LLVM (Oz)",
                marker_color="rgba(128, 128, 128, 0.7)",
                showlegend=False,
                histnorm="probability",
                **hist_kwargs,
            ),
            row=current_row,
            col=1,
        )

        fig.add_vline(
            x=q2,
            line_width=2,
            line_dash="solid",
            line_color="black",
            row=current_row,
            col=1,
        )
        fig.add_vline(
            x=q1,
            line_width=1,
            line_dash="dash",
            line_color="rgba(0, 0, 0, 0.6)",
            row=current_row,
            col=1,
        )
        fig.add_vline(
            x=q3,
            line_width=1,
            line_dash="dash",
            line_color="rgba(0, 0, 0, 0.6)",
            row=current_row,
            col=1,
        )

        fig.update_yaxes(title_text="LLVM (Oz)", row=current_row, col=1)
        current_row += 1

    colors = [
        "rgba(31, 119, 180, 0.6)",
        "rgba(255, 127, 14, 0.6)",
        "rgba(44, 160, 44, 0.6)",
        "rgba(214, 39, 40, 0.6)",
        "rgba(148, 103, 189, 0.6)",
    ]

    for i, (ckpt_dir, model_name) in enumerate(checkpoint_dirs):
        agent_path = os.path.join(ckpt_dir, f"agent_{value_type}.npy")
        agent_values = np.load(agent_path)

        if diff == "llvm_minus_agent":
            agent_values = llvm_values - agent_values
        elif diff == "agent_minus_llvm":
            agent_values = agent_values - llvm_values
            # agent_values = agent_values[agent_values != 0]

        # if model_name == "PPO sparse reward Best":
        #     print(agent_values)

        agent_mean = np.mean(agent_values)
        agent_median = np.median(agent_values)

        color = colors[i % len(colors)]

        min_5_quartile = min(min_5_quartile, np.percentile(agent_values, 5))
        max_95_quartile = max(max_95_quartile, np.percentile(agent_values, 95))
        q1, q2, q3 = np.percentile(agent_values, [25, 50, 75])
        success_rate = np.mean(agent_values >= 0) * 100
        succes_rate_no_draw = np.sum(agent_values > 0) / np.sum(agent_values != 0) * 100
        zeros_rate = np.mean(agent_values == 0) * 100

        logger.info(
            f"Model: {model_name}, Mean: {agent_mean:.2f}, Median: {agent_median:.2f}, Q1: {q1:.2f}, Q3: {q3:.2f}, Success Rate: {success_rate:.2f}%, Success Rate (no draw): {succes_rate_no_draw:.2f}%, Draw Rate: {zeros_rate:.2f}%"
        )

        fig.add_trace(
            go.Histogram(
                x=agent_values,
                name=model_name,
                marker_color=color,
                histnorm="probability",
                showlegend=False,
                **hist_kwargs,
            ),
            row=current_row,
            col=1,
        )

        fig.add_vline(
            x=q2,
            line_width=2,
            line_dash="solid",
            line_color="black",
            row=current_row,
            annotation_text="Q2",
            annotation_position="top right",
            col=1,
        )
        fig.add_vline(
            x=q1,
            line_width=1,
            line_dash="dash",
            line_color="rgba(0, 0, 0, 0.6)",
            annotation_text="Q1",
            annotation_position="top right",
            row=current_row,
            col=1,
        )
        fig.add_vline(
            x=q3,
            line_width=1,
            line_dash="dash",
            line_color="rgba(0, 0, 0, 0.6)",
            annotation_text="Q3",
            annotation_position="top right",
            row=current_row,
            col=1,
        )

        fig.update_yaxes(title_text=model_name, row=current_row, col=1)
        current_row += 1

    dynamic_height = 130 * num_rows

    fig.update_layout(
        title=plot_name,
        template="plotly_white",
        font=dict(size=8),
        margin=dict(l=40, r=40, t=60, b=60),
        height=dynamic_height,
        width=1000,
    )

    fig.update_xaxes(
        range=[min_5_quartile, max_95_quartile] if y_limit is None else y_limit
    )
    fig.update_xaxes(title_text=yaxis_name, row=num_rows, col=1)

    fig.write_image(
        f"plots/{plot_name.replace(' ', '_')}.png", width=600, height=400, scale=2
    )


def calc_perc_gain_opt(checkpoint_dirs: list[tuple[str, str]]):
    logger.info("Calculating percentage gain for optimized runs...")

    first_dir = checkpoint_dirs[0][0]

    llvm_gains_opt = np.load(os.path.join(first_dir, "llvm_gains_opt.npy"))
    size_baseline_opts = np.load("models_final/size_baseline_opts.npy")

    # llvm_perc_gains_opt = (llvm_gains_opt / size_baseline_opts) * 100
    llvm_perc_gains_opt = np.where(
        size_baseline_opts == 0, 0, (llvm_gains_opt / size_baseline_opts) * 100
    )

    q1, q2, q3 = np.percentile(llvm_perc_gains_opt, [25, 50, 75])
    success_rate = np.mean(llvm_perc_gains_opt >= 0) * 100
    no_inlining = np.mean(size_baseline_opts == 0) * 100

    logger.info(
        f"Baseline LLVM (Oz) Percentage Gain (Opt), Mean: {np.mean(llvm_perc_gains_opt):.2f}%, Median: {np.median(llvm_perc_gains_opt):.2f}%, Q1: {q1:.2f}%, Q3: {q3:.2f}%, Success Rate: {success_rate:.2f}%, No Inlining: {no_inlining:.2f}%"
    )

    for ckpt_dir, model_name in checkpoint_dirs:
        agent_gains_opt = np.load(os.path.join(ckpt_dir, "agent_gains_opt.npy"))
        agent_perc_gains_opt = np.where(
            size_baseline_opts == 0, 0, (agent_gains_opt / size_baseline_opts) * 100
        )

        q1, q2, q3 = np.percentile(agent_perc_gains_opt, [25, 50, 75])
        success_rate = np.mean(agent_perc_gains_opt >= 0) * 100
        succes_rate_no_draw = (
            np.sum(agent_perc_gains_opt > 0) / np.sum(agent_perc_gains_opt != 0) * 100
        )
        zeros_rate = np.mean(agent_perc_gains_opt == 0) * 100

        logger.info(
            f"Model: {model_name}, Percentage Gain (Opt), Mean: {np.mean(agent_perc_gains_opt):.2f}%, Median: {np.median(agent_perc_gains_opt):.2f}%, Q1: {q1:.2f}%, Q3: {q3:.2f}%, Success Rate: {success_rate:.2f}%, Success Rate (no draw): {succes_rate_no_draw:.2f}%, Draw Rate: {zeros_rate:.2f}%"
        )

    for ckpt_dir, model_name in checkpoint_dirs:
        agent_gains_opt = np.load(os.path.join(ckpt_dir, "agent_gains_opt.npy"))
        agent_perc_gains_opt = np.where(
            size_baseline_opts == 0, 0, (agent_gains_opt / size_baseline_opts) * 100
        )

        agent_minus_llvm_perc_gain = agent_perc_gains_opt - llvm_perc_gains_opt

        q1, q2, q3 = np.percentile(agent_minus_llvm_perc_gain, [25, 50, 75])
        success_rate = np.mean(agent_minus_llvm_perc_gain >= 0) * 100
        succes_rate_no_draw = (
            np.sum(agent_minus_llvm_perc_gain > 0)
            / np.sum(agent_minus_llvm_perc_gain != 0)
            * 100
        )
        zeros_rate = np.mean(agent_minus_llvm_perc_gain == 0) * 100

        logger.info(
            f"Model: {model_name}, Percentage Gain vs LLVM (Opt), Mean: {np.mean(agent_minus_llvm_perc_gain):.2f}%, Median: {np.median(agent_minus_llvm_perc_gain):.2f}%, Q1: {q1:.2f}%, Q3: {q3:.2f}%, Success Rate: {success_rate:.2f}%, Success Rate (no draw): {succes_rate_no_draw:.2f}%, Draw Rate: {zeros_rate:.2f}%"
        )


def plot_benchmark_results(checkpoint_dirs: list[tuple[str, str]]):
    plot_benchmark_for_metric(
        checkpoint_dirs,
        value_type="gains_no_opt",
        plot_name="Size reduction",
        yaxis_name="Size reduction (bytes)",
        y_limit=(-300, 3000),
    )

    plot_benchmark_for_metric(
        checkpoint_dirs,
        value_type="gains_no_opt",
        plot_name="Size reduction (vs LLVM)",
        yaxis_name="Size reduction (bytes)",
        diff="agent_minus_llvm",
        y_limit=(-300, 300),
        bin_width=10,
    )

    plot_benchmark_for_metric(
        checkpoint_dirs,
        value_type="gains_opt",
        plot_name="Size reduction (excluding other passes)",
        yaxis_name="Size reduction (bytes)",
        y_limit=(-200, 200),
    )

    plot_benchmark_for_metric(
        checkpoint_dirs,
        value_type="gains_opt",
        plot_name="Size reduction (excluding other passes, vs LLVM)",
        yaxis_name="Size reduction (bytes)",
        diff="agent_minus_llvm",
        y_limit=(-200, 200),
    )

    plot_benchmark_for_metric(
        checkpoint_dirs,
        value_type="perc_gains",
        plot_name="Percentage size reduction",
        yaxis_name="Percentage size reduction (%)",
    )

    plot_benchmark_for_metric(
        checkpoint_dirs,
        value_type="perc_gains",
        plot_name="Percentage size reduction (vs LLVM)",
        yaxis_name="Percentage size reduction (%)",
        diff="agent_minus_llvm",
        y_limit=(-40, 40),
    )

    calc_perc_gain_opt(checkpoint_dirs)


if __name__ == "__main__":
    # checkpoints_to_compare = [
    #     ("./models_final/ppo/paper_ppo_128_3", "PPO hidden=128"),
    #     ("./models_final/ppo/paper_ppo_128_best", "PPO hidden=128 Best"),
    #     ("./models_final/ppo/paper_ppo_256_3", "PPO hidden=256"),
    #     ("./models_final/ppo/paper_ppo_256_best", "PPO hidden=256 Best"),
    #     ("./models_final/ppo/paper_ppo_1024_3", "PPO hidden=1024"),
    #     ("./models_final/ppo/paper_ppo_1024_best", "PPO hidden=1024 Best"),
    #     ("./models_final/ppo/paper_ppo_sparse_3", "PPO Sparse Reward"),
    #     ("./models_final/ppo/paper_ppo_sparse_3_best", "PPO Sparse Reward Best"),
    # ]

    # checkpoints_to_compare = [
    #     ("./models_final/ppo/paper_ppo_gat_3", "PPO + GAT"),
    #     ("./models_final/ppo/paper_ppo_gat_contr", "PPO + GAT Contrastive"),
    #     ("./models_final/ppo/paper_ppo_gat_contr_best", "PPO + GAT Contrastive Best"),
    #     ("./models_final/ppo/paper_ppo_gat_pretrain", "PPO + GAT Pretrained"),
    #     ("./models_final/ppo/paper_ppo_gat_pretrain_best", "PPO + GAT Pretrained Best"),
    #     ("./models_final/ppo/paper_ppo_gat_pretrain_noretain", "PPO + GAT Pretrained, no Retain"),
    #     ("./models_final/ppo/paper_ppo_gat_pretrain_noretain_best", "PPO + GAT Pretrained, no Retain Best"),
    #     ("./models_final/ppo/paper_ppo_gat_sparse_3", "PPO + GAT Sparse Reward"),
    #     ("./models_final/ppo/paper_ppo_gat_sparse_3_best", "PPO + GAT Sparse Reward Best"),
    # ]

    # checkpoints_to_compare = [
    #     ("./models_final/ppo/paper_ppo_sage_3", "PPO + GraphSAGE"),
    #     ("./models_final/ppo/paper_ppo_sage_3_best", "PPO + GraphSAGE Best"),
    #     ("./models_final/ppo/paper_ppo_sage_contr", "PPO + GraphSAGE Contrastive"),
    #     ("./models_final/ppo/paper_ppo_sage_contr_best", "PPO + GraphSAGE Contrastive Best"),
    #     ("./models_final/ppo/paper_ppo_sage_pretrain", "PPO + GraphSAGE Pretrained"),
    #     ("./models_final/ppo/paper_ppo_sage_pretrain_best", "PPO + GraphSAGE Pretrained Best"),
    #     # ("./models_final/ppo/paper_ppo_sage_pretrain_noretain", "PPO + GraphSAGE Pretrained, no Retain"),
    #     # ("./models_final/ppo/paper_ppo_sage_pretrain_noretain_best", "PPO + GraphSAGE Pretrained, no Retain Best"),
    #     # ("./models_final/ppo/paper_ppo_sage_sparse_4", "PPO + GraphSAGE Sparse Reward"),
    #     # ("./models_final/ppo/paper_ppo_sage_sparse_4_best", "PPO + GraphSAGE Sparse Reward Best"),
    # ]

    # checkpoints_to_compare = [
    #     ("./models_final/ppo/paper_ppo_gat_3", "PPO + GAT"),
    #     ("./models_final/ppo/paper_ppo_gat_sparse_3_best", "PPO + GAT Sparse Reward Best"),
    #     ("./models_final/ppo/paper_ppo_sage_3_best", "PPO + GraphSAGE Best"),
    #     ("./models_final/ppo/paper_ppo_sage_sparse_4_best", "PPO + GraphSAGE Sparse Reward Best"),
    # ]

    # checkpoints_to_compare = [
    #     ("./models_final/ppo/paper_ppo_gat_pretrain_noretain_best", "PPO + GAT Pretrained, no Retain Best"),
    #     ("./models_final/ppo/paper_ppo_sage_pretrain_noretain_best", "PPO + GraphSAGE Pretrained, no Retain Best"),

    #     # ("./models_final/ppo/paper_ppo_gat_contr_best", "PPO + GAT Contrastive Best"),
    #     ("./models_final/ppo/paper_ppo_gat_pretrain_best", "PPO + GAT Pretrained Best"),
    #     # ("./models_final/ppo/paper_ppo_sage_contr_best", "PPO + GraphSAGE Contrastive Best"),
    #     ("./models_final/ppo/paper_ppo_sage_pretrain_best", "PPO + GraphSAGE Pretrained Best"),
    # ]

    checkpoints_to_compare = [
        # ("./models_final/ppo/paper_ppo_128_best", "PPO hidden=128 Best"),
        # ("./models_final/ppo/paper_ppo_256_best", "PPO hidden=256 Best"),
        # ("./models_final/ppo/paper_ppo_1024_best", "PPO hidden=1024 Best"),
        ("./models_final/ppo/paper_ppo_sparse_3_best", "PPO (Sparse)"),
        # ("./models_final/ppo/paper_ppo_gat_3", "PPO + GAT"),
        # (
        #     "./models_final/ppo/paper_ppo_gat_sparse_3_best",
        #     "PPO + GAT Sparse Reward Best",
        # ),
        # ("./models_final/ppo/paper_ppo_gat_contr_best", "PPO + GAT Contrastive Best"),
        # ("./models_final/ppo/paper_ppo_gat_pretrain_best", "PPO + GAT Pretrained Best"),
        # (
        #     "./models_final/ppo/paper_ppo_gat_pretrain_noretain_best",
        #     "PPO + GAT Pretrained, no Retain Best",
        # ),
        # ("./models_final/ppo/paper_ppo_sage_3_best", "PPO + GraphSAGE Best"),
        # (
        #     "./models_final/ppo/paper_ppo_sage_sparse_4_best",
        #     "PPO + GraphSAGE Sparse Reward Best",
        # ),
        # (
        #     "./models_final/ppo/paper_ppo_sage_contr_best",
        #     "PPO + GraphSAGE Contrastive Best",
        # ),
        # (
        #     "./models_final/ppo/paper_ppo_sage_pretrain_best",
        #     "PPO + GraphSAGE Pretrained Best",
        # ),
        # (
        #     "./models_final/ppo/paper_ppo_sage_pretrain_noretain_best",
        #     "PPO + GraphSAGE Pretrained, no Retain Best",
        # ),
    ]

    plot_benchmark_results(checkpoints_to_compare)
