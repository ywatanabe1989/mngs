#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-26 08:38:30 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/ax/_raster_plot.py


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
from bisect import bisect_left
from glob import glob
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from natsort import natsorted
from tqdm import tqdm

"""
Warnings
"""
warnings.simplefilter("ignore", UserWarning)


"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def raster_plot(ax, positions, time=None, **kwargs):
    """
    Create a raster plot using eventplot and return the plot along with a DataFrame.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the raster plot.
    positions : list of lists or array-like
        Position of events for each channel. Each list corresponds to events of one channel.
    time : array-like, optional
        The time indices for the events. If None, time will be generated based on event positions.
    **kwargs : keyword arguments
        Additional keyword arguments to be passed to the eventplot function.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes with the raster plot.
    df : pandas.DataFrame
        A DataFrame where rows correspond to time indices and columns correspond to channels.
        Each cell contains the channel index for events at specific time indices.

    """

    def positions_to_df(positions, time):
        if time is None:
            time = np.linspace(0, np.hstack(positions).max(), 1000)

        digi = np.nan * np.zeros([len(positions), len(time)], dtype=int)

        for i_unit, positions_unit in enumerate(positions):

            if isinstance(positions_unit, [float, int]):
                positions_unit = [positions_unit]

            for pu in positions_unit:

                i_ins = bisect_left(time, pu)
                if i_ins == digi.shape[-1]:
                    i_ins -= i_unit
                digi[i_unit, i_ins] = i_unit

        return pd.DataFrame(np.array(digi)).T.set_index(time)

    def ensure_list(positions):
        out = []
        for pu in positions:
            if isinstance(pu, list):
                out.append(pu)
            elif isinstance(pu, (int, float)):
                out.append([pu])
        return out

    positions = ensure_list(positions)

    df = positions_to_df(positions, time)

    # Use eventplot to create a raster plot
    ax.eventplot(positions, orientation="horizontal", **kwargs)

    # # Set labels for axes
    # ax.set_xlabel("Time Index")
    # ax.set_ylabel("Channel Index")

    # # Set y-ticks to correspond to each channel
    # ax.set_yticks(range(len(df.columns)))
    # ax.set_yticklabels(df.columns)

    return ax, df


def test():
    # Example positions data
    positions = [
        [10, 50, 90],  # Channel 0
        [20, 60, 100],  # Channel 1
        [30, 70, 110],  # Channel 2
        [40, 80, 120],  # Channel 3
    ]

    # Create a new figure and axes
    fig, ax = mngs.plt.subplots()

    # Call the raster function
    ax, df = raster_plot(ax, positions)


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
    test()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
