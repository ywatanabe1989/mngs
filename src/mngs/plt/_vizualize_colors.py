#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 21:20:46 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_vizualize_colors.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_vizualize_colors.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import sys

import matplotlib.pyplot as plt
import numpy as np


def vizualize_colors(colors):

    def gen_rand_sample(size=100):
        x = np.linspace(-1, 1, size)
        y = np.random.normal(size=size)
        s = np.random.randn(size)
        return x, y, s

    from . import subplots as mngs_plt_subplots

    fig, axes = mngs_plt_subplots(ncols=4)

    for ii, (color_str, rgba) in enumerate(colors.items()):
        xx, yy, ss = gen_rand_sample()

        # Box color plot
        axes[0].rectangle(xx=ii, yy=0, ww=1, hh=1, color=rgba, label=color_str)

        # Line plot
        axes[1].plot_with_ci(xx, yy, ss, color=rgba, label=color_str)

        # Scatter plot
        axes[2].scatter(xx, yy, color=rgba, label=color_str)

        # KDE plot
        axes[3].kde(yy, color=rgba, label=color_str)

    for ax in axes.flat:
        # ax.axis("off")
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import mngs

    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()
    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF