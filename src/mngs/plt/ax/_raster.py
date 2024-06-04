#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-03 14:45:57 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/ax/_raster.py


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


def raster(ax, positions, time=None, **kwargs):
    def positions_to_df(positions, time):
        if time is None:
            time = np.arange(0, np.hstack(positions).max(), 1000)

        digi = np.nan * np.zeros([len(positions), len(time)], dtype=int)

        for i_unit, positions_unit in enumerate(positions):

            for pu in positions_unit:
                i_ins = bisect_left(time, pu)
                if i_ins == digi.shape[-1]:
                    i_ins -= i_unit
                digi[i_unit, i_ins] = i_unit

        return pd.DataFrame(np.array(digi)).T.set_index(time)

    df = positions_to_df(positions, time)

    # Use eventplot to create a raster plot
    ax.eventplot(positions, orientation="horizontal", **kwargs)

    # Set labels for axes
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Channel Index")

    # Set y-ticks to correspond to each channel
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)

    return ax, df


def main():
    pass


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
