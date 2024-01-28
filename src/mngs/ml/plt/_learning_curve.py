#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-26 16:17:42 (ywatanabe)"

import re

import matplotlib
import matplotlib.pyplot as plt
import mngs
import pandas as pd


def learning_curve(
    metrics_df,
    keys_to_plot=["loss"],
    title="Title",
    max_n_ticks=4,
    scattersize=3,
    linewidth=1,
    yscale="linear",
):
    COLORS_DICT = {
        "Training": "blue",
        "Validation": "green",
        "Test": "red",
    }

    fig, axes = plt.subplots(len(keys_to_plot), 1, sharex=True, sharey=False)

    axes = axes if len(keys_to_plot) != 1 else [axes]

    axes[-1].set_xlabel("Iteration#")
    fig.text(0.5, 0.95, title, ha="center")

    for i_plt, key_plt in enumerate(keys_to_plot):
        ax = axes[i_plt]
        ax.set_yscale(yscale)
        # ax.set_ylabel(self._rename_if_key_to_plot(key_plt))
        ax.set_ylabel(key_plt)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(max_n_ticks))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(max_n_ticks))

        if re.search("[aA][cC][cC]", key_plt):  # acc, ylim, yticks
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1.0])

        # for step in self.dfs_pivot_i_global.keys():
        for step in COLORS_DICT.keys():

            # if (step == re.search("^[Tt]rain", step)) or (step == re.search("^[Tt]raining", step)):
            if re.match("^[Tt]rain(ing)?", step):  # [REVISED]
                ax.plot(
                    metrics_df[metrics_df.step == step].index,
                    metrics_df[metrics_df.step == step][key_plt],
                    label=step,
                    color=mngs.plt.colors.to_RGBA(
                        COLORS_DICT[step], alpha=0.9
                    ),
                    linewidth=linewidth,
                )
                ax.legend()
                ########################################
                ## Epoch starts points; just in "Training" not to b duplicated
                ########################################
                epoch_starts = abs(
                    metrics_df[metrics_df.step == step]["i_epoch"]
                    - metrics_df[metrics_df.step == step]["i_epoch"].shift(-1)
                )
                indi_global_epoch_starts = [0] + list(
                    epoch_starts[epoch_starts == 1].index
                )

                for i_epoch, i_global_epoch_start in enumerate(
                    indi_global_epoch_starts
                ):
                    ax.axvline(
                        x=i_global_epoch_start,
                        ymin=-1e4,  # ax.get_ylim()[0],
                        ymax=1e4,  # ax.get_ylim()[1],
                        linestyle="--",
                        color=mngs.plt.colors.to_RGBA("gray", alpha=0.1),
                    )
                ########################################

            if (step == "Validation") or (step == "Test"):  # scatter
                ax.scatter(
                    metrics_df[metrics_df.step == step].index,
                    metrics_df[metrics_df.step == step][key_plt],
                    label=step,
                    color=mngs.plt.colors.to_RGBA(
                        COLORS_DICT[step], alpha=0.9
                    ),
                    s=scattersize,
                    alpha=0.9,
                )
                ax.legend()

    return fig


if __name__ == "__main__":
    sdir = "./scripts/train_EEGPT/2024-0126-09:25_bvYp7bIr/"
    metrics = mngs.io.load(sdir + "metrics.csv")
    # del metrics["Unnamed: 0"]
    metrics = metrics.set_index("i_global")
    fig = learning_curve(metrics, yscale="log")

    mngs.io.save(fig, sdir + "learning_curve.png")
