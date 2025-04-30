#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 21:01:53 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_ecdf.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/ax/_ecdf.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import numpy as np

from ....pd._force_df import force_df as mngs_pd_force_df


def plot_ecdf(axis, data, **kwargs):
    """Plot Empirical Cumulative Distribution Function (PLOT_ECDF).

    The PLOT_ECDF shows the proportion of data points less than or equal to each value,
    representing the empirical estimate of the cumulative distribution function.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Matplotlib axis to plot on
    data : array-like
        Data to compute and plot PLOT_ECDF for
    **kwargs : dict
        Additional arguments to pass to plot function

    Returns
    -------
    tuple
        (axis, DataFrame) containing the plot and data
    """
    assert isinstance(
        axis, matplotlib.axes._axes.Axes
    ), "First argument must be a matplotlib axis"

    # Flatten and remove NaN values
    data = np.hstack(data)
    data = data[~np.isnan(data)]
    nn = len(data)

    # Sort the data and compute the PLOT_ECDF values
    data_sorted = np.sort(data)
    plot_ecdf_perc = (
        100 * np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    )

    # Create the pseudo x-axis for step plotting
    x_step = np.repeat(data_sorted, 2)[1:]
    y_step = np.repeat(plot_ecdf_perc, 2)[:-1]

    # Plot the PLOT_ECDF using steps
    axis.plot(x_step, y_step, drawstyle="steps-post", **kwargs)

    # Scatter the original data points
    axis.plot(data_sorted, plot_ecdf_perc, marker=".", linestyle="none")

    # Set ylim, xlim, and aspect ratio
    axis.set_ylim(0, 100)
    axis.set_xlim(0, 1.0)

    # Create a DataFrame to hold the PLOT_ECDF data
    df = mngs_pd_force_df(
        {
            "x": data_sorted,
            "y": plot_ecdf_perc,
            "n": nn,
            "x_step": x_step,
            "y_step": y_step,
        }
    )

    return axis, df

# EOF