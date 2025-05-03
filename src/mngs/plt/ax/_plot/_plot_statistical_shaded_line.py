#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 20:50:45 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_plot_statistical_shaded_line.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/ax/_plot_statistical_shaded_line.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import numpy as np
import pandas as pd

from ._plot_shaded_line import plot_shaded_line as mngs_plt_plot_shaded_line


def plot_line(axis, data, xx=None, **kwargs):
    """Plot a simple line."""
    assert isinstance(
        axis, matplotlib.axes._axes.Axes
    ), "First argument must be a matplotlib axis"
    data = np.asarray(data)
    assert data.ndim <= 2, f"Data must be 1D or 2D, got {data.ndim}D"
    if xx is None:
        xx = np.arange(len(data))
    else:
        xx = np.asarray(xx)
    assert len(xx) == len(
        data
    ), f"xx length ({len(xx)}) must match data length ({len(data)})"
    axis.plot(xx, data, **kwargs)
    return axis, pd.DataFrame({"x": xx, "y": data})


def plot_mean_std(axis, data, xx=None, sd=1, **kwargs):
    """Plot mean line with standard deviation shading."""
    assert isinstance(
        axis, matplotlib.axes._axes.Axes
    ), "First argument must be a matplotlib axis"
    assert isinstance(sd, (int, float)), f"sd must be a number, got {type(sd)}"
    assert sd >= 0, f"sd must be non-negative, got {sd}"
    data = np.asarray(data)
    assert data.ndim <= 2, f"Data must be 1D or 2D, got {data.ndim}D"
    if xx is None:
        xx = np.arange(data.shape[1] if data.ndim > 1 else len(data))
    else:
        xx = np.asarray(xx)
    expected_len = data.shape[1] if data.ndim > 1 else len(data)
    assert (
        len(xx) == expected_len
    ), f"xx length ({len(xx)}) must match data length ({expected_len})"

    if data.ndim == 1:
        central = data
        error = np.zeros_like(central)
    else:
        central = np.nanmean(data, axis=0)
        error = np.nanstd(data, axis=0) * sd

    y_lower = central - error
    y_upper = central + error
    n_samples = data.shape[0] if data.ndim > 1 else 1

    if "label" in kwargs and kwargs["label"]:
        kwargs["label"] = f"{kwargs['label']} (n={n_samples})"

    return mngs_plt_plot_shaded_line(
        axis, xx, y_lower, central, y_upper, **kwargs
    )


def plot_mean_ci(axis, data, xx=None, perc=95, **kwargs):
    """Plot mean line with confidence interval shading."""
    assert isinstance(
        axis, matplotlib.axes._axes.Axes
    ), "First argument must be a matplotlib axis"
    assert isinstance(
        perc, (int, float)
    ), f"perc must be a number, got {type(perc)}"
    assert 0 <= perc <= 100, f"perc must be between 0 and 100, got {perc}"
    data = np.asarray(data)
    assert data.ndim <= 2, f"Data must be 1D or 2D, got {data.ndim}D"

    if xx is None:
        xx = np.arange(data.shape[1] if data.ndim > 1 else len(data))
    else:
        xx = np.asarray(xx)

    expected_len = data.shape[1] if data.ndim > 1 else len(data)
    assert (
        len(xx) == expected_len
    ), f"xx length ({len(xx)}) must match data length ({expected_len})"

    if data.ndim == 1:
        central = data
        y_lower = central
        y_upper = central
    else:
        central = np.nanmean(data, axis=0)
        # Calculate CI bounds
        alpha = 1 - perc / 100
        y_lower_perc = alpha / 2 * 100
        y_upper_perc = (1 - alpha / 2) * 100
        y_lower = np.nanpercentile(data, y_lower_perc, axis=0)
        y_upper = np.nanpercentile(data, y_upper_perc, axis=0)

    n_samples = data.shape[0] if data.ndim > 1 else 1

    if "label" in kwargs and kwargs["label"]:
        kwargs["label"] = f"{kwargs['label']} (n={n_samples}, CI={perc}%)"

    return mngs_plt_plot_shaded_line(
        axis, xx, y_lower, central, y_upper, **kwargs
    )


def plot_median_iqr(axis, data, xx=None, **kwargs):
    """Plot median line with interquartile range shading."""
    assert isinstance(
        axis, matplotlib.axes._axes.Axes
    ), "First argument must be a matplotlib axis"
    data = np.asarray(data)
    assert data.ndim <= 2, f"Data must be 1D or 2D, got {data.ndim}D"

    if xx is None:
        xx = np.arange(data.shape[1] if data.ndim > 1 else len(data))
    else:
        xx = np.asarray(xx)

    expected_len = data.shape[1] if data.ndim > 1 else len(data)
    assert (
        len(xx) == expected_len
    ), f"xx length ({len(xx)}) must match data length ({expected_len})"

    if data.ndim == 1:
        central = data
        y_lower = central
        y_upper = central
    else:
        central = np.nanmedian(data, axis=0)
        y_lower = np.nanpercentile(data, 25, axis=0)
        y_upper = np.nanpercentile(data, 75, axis=0)

    n_samples = data.shape[0] if data.ndim > 1 else 1

    if "label" in kwargs and kwargs["label"]:
        kwargs["label"] = f"{kwargs['label']} (n={n_samples}, IQR)"

    return mngs_plt_plot_shaded_line(
        axis, xx, y_lower, central, y_upper, **kwargs
    )

# EOF