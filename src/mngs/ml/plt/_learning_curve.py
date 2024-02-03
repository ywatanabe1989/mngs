#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-02 09:25:07 (ywatanabe)"

import re

import matplotlib
import matplotlib.pyplot as plt
import mngs
import pandas as pd


def process_i_global(metrics_df):
    if metrics_df.index.name != "i_global":
        try:
            metrics_df = metrics_df.set_index("i_global")
        except KeyError:
            print(
                "Error: The DataFrame does not contain a column named 'i_global'. Please check the column names."
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        print("The index is already set to 'i_global'.")
    metrics_df["i_global"] = metrics_df.index  # alias
    return metrics_df


def set_yaxis_for_acc(ax, key_plt):
    if re.search("[aA][cC][cC]", key_plt):  # acc, ylim, yticks
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1.0])
    return ax


def plot_tra(ax, metrics_df, key_plt, lw=1):
    indi_step = mngs.gen.search(
        "^[Tt]rain(ing)?", metrics_df.step, as_bool=True
    )[0]
    step_df = metrics_df[indi_step]

    if len(step_df) != 0:
        ax.plot(
            step_df.index,  # i_global
            step_df[key_plt],
            label="Training",
            # color=COLORS_DICT["tra"],
            color=mngs.plt.colors.to_RGBA("blue", alpha=0.9),
            linewidth=lw,
        )
        ax.legend()

    return ax


def scatter_val(ax, metrics_df, key_plt, s=3):
    indi_step = mngs.gen.search(
        "^[Vv]alid(ation)?", metrics_df.step, as_bool=True
    )[0]
    step_df = metrics_df[indi_step]
    if len(step_df) != 0:
        ax.scatter(
            step_df.index,
            step_df[key_plt],
            label="Validation",
            color=mngs.plt.colors.to_RGBA("green", alpha=0.9),
            s=s,
            alpha=0.9,
        )
        ax.legend()
    return ax


def scatter_tes(ax, metrics_df, key_plt, s=3):
    indi_step = mngs.gen.search("^[Tt]est", metrics_df.step, as_bool=True)[0]
    step_df = metrics_df[indi_step]
    if len(step_df) != 0:
        ax.scatter(
            step_df.index,
            step_df[key_plt],
            label="Test",
            color=mngs.plt.colors.to_RGBA("red", alpha=0.9),
            # color=COLORS_DICT["tes"],
            s=s,
            alpha=0.9,
        )
        ax.legend()
    return ax


def vline_at_epochs(ax, metrics_df):
    # Determine the global iteration values where new epochs start
    epoch_starts = metrics_df[metrics_df["i_batch"] == 0].index.values
    epoch_labels = metrics_df[metrics_df["i_batch"] == 0].index.values
    ax.vlines(
        x=epoch_starts,
        ymin=-1e4,  # ax.get_ylim()[0],
        ymax=1e4,  # ax.get_ylim()[1],
        linestyle="--",
        color=mngs.plt.colors.to_RGBA("gray", alpha=0.1),
    )
    return ax


def select_ticks(metrics_df, max_n_ticks=4):
    # Calculate epoch starts and their corresponding labels for ticks
    unique_epochs = metrics_df["i_epoch"].drop_duplicates().values
    epoch_starts = (
        metrics_df[metrics_df["i_batch"] == 0]["i_global"]
        .drop_duplicates()
        .values
    )

    # Given the performance issue, let's just select a few epoch starts for labeling
    # We use MaxNLocator to pick ticks; however, it's used here to choose a reasonable number of epoch markers
    if len(epoch_starts) > max_n_ticks:
        selected_ticks = np.linspace(
            epoch_starts[0], epoch_starts[-1], max_n_ticks, dtype=int
        )
        # Ensure selected ticks are within the epoch starts for accurate labeling
        selected_labels = [
            metrics_df[metrics_df["i_global"] == tick]["i_epoch"].iloc[0]
            for tick in selected_ticks
        ]
    else:
        selected_ticks = epoch_starts
        selected_labels = unique_epochs
    return selected_ticks, selected_labels


def learning_curve(
    metrics_df,
    keys_to_plot=["loss"],
    title="Title",
    max_n_ticks=4,
    scattersize=3,
    linewidth=1,
    yscale="linear",
):
    metrics_df = process_i_global(metrics_df)
    selected_ticks, selected_labels = select_ticks(metrics_df)

    fig, axes = plt.subplots(len(keys_to_plot), 1, sharex=True, sharey=False)
    axes = axes if len(keys_to_plot) != 1 else [axes]

    # axes[-1].set_xlabel("Iteration#")
    axes[-1].set_xlabel("Epoch #")
    fig.text(0.5, 0.95, title, ha="center")

    for i_plt, key_plt in enumerate(keys_to_plot):
        ax = axes[i_plt]
        ax.set_yscale(yscale)
        ax.set_ylabel(key_plt)
        ax = mngs.plt.ax_set_n_ticks(ax)
        ax = set_yaxis_for_acc(ax, key_plt)
        ax = plot_tra(ax, metrics_df, key_plt, lw=linewidth)
        ax = scatter_val(ax, metrics_df, key_plt, s=scattersize)
        ax = scatter_tes(ax, metrics_df, key_plt, s=scattersize)
        # ax = vline_at_epochs(ax, metrics_df)

        # Custom tick marks
        ax = mngs.plt.ax_set_n_ticks(ax)
        ax = mngs.plt.ax_map_ticks(
            ax, selected_ticks, selected_labels, axis="x"
        )

        # # ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=max_n_ticks))
        # # ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=max_n_ticks))
        # # Set custom tick positions and labels to reflect selected epoch starts
        # ax.set_xticks(selected_ticks)
        # ax.set_xticklabels(selected_labels, rotation=45, ha="right")

    return fig


if __name__ == "__main__":

    lpath = "./scripts/train_EEGPT/2024-01-29-12-04_eDflsnWv_v8/metrics.csv"
    sdir, _, _ = mngs.gen.split_fpath(lpath)
    # sdir = "./scripts/train_EEGPT/[DEBUG] 2024-01-29-07-27_A5HS3f0e/"
    metrics_df = mngs.io.load(lpath)
    fig = learning_curve(
        metrics_df, title="Pretraining on db_v8", yscale="log"
    )
    # plt.show()
    mngs.io.save(fig, sdir + "learning_curve.png")
