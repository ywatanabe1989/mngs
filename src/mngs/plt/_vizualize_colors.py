#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-10 10:33:54 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_check_colors.py


"""
This script does XYZ.
"""


"""
Imports
"""
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

import mngs

importlib.reload(mngs)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from natsort import natsorted
from glob import glob
from pprint import pprint
import warnings
import logging
from tqdm import tqdm
import xarray as xr

# sys.path = ["."] + sys.path
# from scripts import utils, load

"""
Warnings
"""
# warnings.simplefilter("ignore", UserWarning)


"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def main():
    vizualize_colors(CC)
    pass


def vizualize_colors(colors):

    def gen_rand_sample(size=100):
        x = np.linspace(-1, 1, size)
        y = np.random.normal(size=size)
        s = np.random.randn(size)
        return x, y, s

    fig, axes = mngs.plt.subplots(ncols=4)

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
