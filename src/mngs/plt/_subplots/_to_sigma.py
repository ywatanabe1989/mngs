#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-10 12:40:03 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/_to_sigma.py


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
import mngs
import seaborn as sns

mngs.gen.reload(mngs)
import warnings
from glob import glob
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from icecream import ic
from natsort import natsorted
from tqdm import tqdm

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
import mngs
import pandas as pd


def to_sigma(history):
    try:
        if len(history) > 0:
            data_frames = [
                format_plotting_args(record) for record in history.values()
            ]
            combined_data = pd.concat(data_frames, join="inner", axis=1)
            return combined_data.apply(
                lambda col: col.dropna().reset_index(drop=True)
            )
        else:
            return
    except Exception as e:
        __import__("ipdb").set_trace()
        print(e)


def format_plotting_args(record):
    id, method, args, kwargs = record

    # Matplotlib
    if method in ["plot", "scatter", "bar"]:
        x, y = args
        df = pd.DataFrame(
            {f"{id}_{method}_x": x, f"{id}_{method}_y": y}
        )  # .apply(lambda col: col.dropna().reset_index(drop=True))
        return df

    elif method == "boxplot":
        xs = args[0]

        # One box plot
        if mngs.gen.is_listed_X(xs, [float, int]):
            df = pd.DataFrame(xs)
        # Multiple boxes
        else:
            df = mngs.gen.force_dataframe(
                {i_x: _x for i_x, _x in enumerate(xs)}
            )
        df.columns = [f"{id}_{method}_{col}_x" for col in df.columns]
        df = df.apply(lambda col: col.dropna().reset_index(drop=True))
        return df

    # Original
    elif method == "plot_with_ci":
        xx, mm, ss = args
        df = pd.DataFrame(
            {
                f"{id}_{method}_x": xx,
                f"{id}_{method}_under": mm - ss,
                f"{id}_{method}_mean": mm,
                f"{id}_{method}_upper": mm + ss,
            }
        )
        return df

    elif method == "raster":
        df = args[0]
        return df

    elif method == "sns_barplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "sns_boxplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "sns_heatmap":
        __import__("ipdb").set_trace()
        return df

    elif method == "sns_histplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "sns_kdeplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "sns_lineplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "sns_pairplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "sns_scatterplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "sns_violinplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "sns_jointplot":
        __import__("ipdb").set_trace()
        return df

    else:
        raise NotImplementedError(f"{method} is not implemented.")


def main():
    # Line
    fig, ax = mngs.plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6], id="plot1")
    ax.plot([4, 5, 6], [1, 2, 3], id="plot2")
    mngs.io.save(fig, "./plots.png")
    mngs.io.save(ax.to_sigma(), "./plots.csv")

    # No tracking
    fig, ax = mngs.plt.subplots(track=False)
    ax.plot([1, 2, 3], [4, 5, 6], id="plot3")
    ax.plot([4, 5, 6], [1, 2, 3], id="plot4")
    mngs.io.save(fig, "./plots_wo_tracking.png")
    mngs.io.save(ax.to_sigma(), "./plots_wo_tracking.csv")

    # Scatter
    fig, ax = mngs.plt.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6], id="scatter1")
    ax.scatter([4, 5, 6], [1, 2, 3], id="scatter2")
    mngs.io.save(fig, "./scatters.png")
    mngs.io.save(ax.to_sigma(), "./scatters.csv")

    # Box
    fig, ax = mngs.plt.subplots()
    ax.boxplot([1, 2, 3], id="boxplot1")
    mngs.io.save(fig, "./boxplot1.png")
    mngs.io.save(ax.to_sigma(), "./boxplot1.csv")

    # Bar
    fig, ax = mngs.plt.subplots()
    ax.bar(["A", "B", "C"], [4, 5, 6], id="bar1")
    mngs.io.save(fig, "./bar1.png")
    mngs.io.save(ax.to_sigma(), "./bar1.csv")

    # # Bar
    # # fig, ax = mngs.plt.subplots()
    # fig, ax = plt.subplots()
    # ax.bar(["A", "B", "C"], [4, 5, 6], id="bar2")
    # mngs.io.save(fig, "./bar2.png")
    # mngs.io.save(ax.to_sigma(), "./bar2.csv")

    # print(ax.to_sigma())
    # #    plot1_plot_x  plot1_plot_y  plot2_plot_x  ...  boxplot1_boxplot_x  bar1_bar_x  bar1_bar_y
    # # 0           1.0           4.0           4.0  ...                 1.0           A         4.0
    # # 1           2.0           5.0           5.0  ...                 2.0           B         5.0
    # # 2           3.0           6.0           6.0  ...                 3.0           C         6.0

    # print(ax.to_sigma().keys())  # plot3 and plot 4 are not tracked
    # # [3 rows x 11 columns]
    # # Index(['plot1_plot_x', 'plot1_plot_y', 'plot2_plot_x', 'plot2_plot_y',
    # #        'scatter1_scatter_x', 'scatter1_scatter_y', 'scatter2_scatter_x',
    # #        'scatter2_scatter_y', 'boxplot1_boxplot_x', 'bar1_bar_x', 'bar1_bar_y'],
    # #       dtype='object')

    # # If a path is passed, the sigmaplot-friendly dataframe is saved as a csv file.
    # ax.to_sigma("../for_sigmaplot.csv")
    # # Saved to: ../for_sigmaplot.csv


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
