#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 15:23:01 (ywatanabe)"
# File: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_plot/_plot_raster.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/ax/_plot/_plot_raster.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib

from bisect import bisect_left

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_raster(
    ax,
    event_times,
    time=None,
    labels=None,
    colors=None,
    orientation="horizontal",
    **kwargs
):
    """
    Create a raster plot using eventplot with custom labels and colors.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the raster plot.
    event_times : Array-like or list of lists
        Time points of events by channels
    time : array-like, optional
        The time indices for the events (default: np.linspace(0, max(event_times))).
    labels : list, optional
        Labels for each channel.
    colors : list, optional
        Colors for each channel.
    orientation: str, optional
        Orientation of raster plot (default: horizontal).
    **kwargs : dict
        Additional keyword arguments for eventplot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the raster plot.
    df : pandas.DataFrame
        DataFrame with time indices and channel events.
    """
    assert isinstance(
        ax, matplotlib.axes._axes.Axes
    ), "First argument must be a matplotlib axis"

    # Format event_times data
    event_times_list = _ensure_list(event_times)

    # Handle colors and labels
    colors = _handle_colors(colors, event_times_list)

    # Plotting as eventplot using event_times_list
    for ii, (pos, color) in enumerate(zip(event_times_list, colors)):
        label = _define_label(labels, ii)
        ax.eventplot(
            pos, orientation=orientation, colors=color, label=label, **kwargs
        )

    # Legend
    if labels is not None:
        ax.legend()

    # Return event_times in a useful format
    event_times_digital_df = _event_times_to_digital_df(event_times_list, time)

    return ax, event_times_digital_df


def _ensure_list(event_times):
    return [
        [pos] if isinstance(pos, (int, float)) else pos for pos in event_times
    ]


def _define_label(labels, ii):
    if (labels is not None) and (ii < len(labels)):
        return labels[ii]
    else:
        return None


def _handle_colors(colors, event_times_list):
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if len(colors) < len(event_times_list):
        colors = colors * (len(event_times_list) // len(colors) + 1)
    return colors


def _event_times_to_digital_df(event_times_list, time):
    if time is None:
        time = np.linspace(
            0, np.max([np.max(pos) for pos in event_times_list]), 1000
        )

    digi = np.full((len(event_times_list), len(time)), np.nan, dtype=float)

    for i_ch, posis_ch in enumerate(event_times_list):
        for posi_ch in posis_ch:
            i_insert = bisect_left(time, posi_ch)
            if i_insert == len(time):
                i_insert -= 1
            digi[i_ch, i_insert] = i_ch

    return pd.DataFrame(digi.T, index=time)

# EOF