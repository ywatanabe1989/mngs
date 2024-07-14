#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-12 20:23:21 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_share_axes.py


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
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def sharex(*multiple_axes):
    xlim = get_global_xlim(*multiple_axes)
    return set_xlims(*multiple_axes, xlim=xlim)


def sharey(*multiple_axes):
    ylim = get_global_ylim(*multiple_axes)
    return set_ylims(*multiple_axes, ylim=ylim)


def get_global_xlim(*multiple_axes):
    xmin, xmax = np.inf, -np.inf
    for axes in multiple_axes:
        # axes
        if isinstance(
            axes, (np.ndarray, mngs.plt._subplots._AxesWrapper.AxesWrapper)
        ):
            for ax in axes.flat:
                _xmin, _xmax = ax.get_ylim()
                xmin = min(xmin, _xmin)
                xmax = max(xmax, _xmax)
        # axis
        else:
            ax = axes
            __import__("ipdb").set_trace()
            _xmin, _xmax = ax.get_ylim()
            xmin = min(xmin, _xmin)
            xmax = max(xmax, _xmax)

    return (xmin, xmax)


# def get_global_xlim(*multiple_axes):
#     xmin, xmax = np.inf, -np.inf
#     for axes in multiple_axes:
#         for ax in axes.flat:
#             _xmin, _xmax = ax.get_xlim()
#             xmin = min(xmin, _xmin)
#             xmax = max(xmax, _xmax)
#     return (xmin, xmax)


def get_global_ylim(*multiple_axes):
    ymin, ymax = np.inf, -np.inf
    for axes in multiple_axes:
        # axes
        if isinstance(
            axes, (np.ndarray, mngs.plt._subplots._AxesWrapper.AxesWrapper)
        ):
            for ax in axes.flat:
                _ymin, _ymax = ax.get_ylim()
                ymin = min(ymin, _ymin)
                ymax = max(ymax, _ymax)
        # axis
        else:
            ax = axes
            _ymin, _ymax = ax.get_ylim()
            ymin = min(ymin, _ymin)
            ymax = max(ymax, _ymax)

    return (ymin, ymax)


def set_xlims(*multiple_axes, xlim=None):
    if xlim is None:
        raise ValueError("Please set xlim. get_global_xlim() might be useful.")

    for axes in multiple_axes:
        # axes
        if isinstance(
            axes, (np.ndarray, mngs.plt._subplots._AxesWrapper.AxesWrapper)
        ):
            for ax in axes.flat:
                ax.set_xlim(xlim)
        # axis
        else:
            ax = axes
            ax.set_xlim(xlim)

    # Return
    if len(multiple_axes) == 1:
        return multiple_axes[0], xlim
    else:
        return multiple_axes, xlim


def set_ylims(*multiple_axes, ylim=None):
    if ylim is None:
        raise ValueError("Please set ylim. get_global_xlim() might be useful.")

    for axes in multiple_axes:
        # axes
        if isinstance(
            axes, (np.ndarray, mngs.plt._subplots._AxesWrapper.AxesWrapper)
        ):
            for ax in axes.flat:
                ax.set_ylim(ylim)

        # axis
        else:
            ax = axes
            ax.set_ylim(ylim)

    # Return
    if len(multiple_axes) == 1:
        return multiple_axes[0], ylim
    else:
        return multiple_axes, ylim


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
