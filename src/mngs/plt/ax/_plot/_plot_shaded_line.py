#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 20:46:58 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_plot_shaded_line.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/ax/_plot_shaded_line.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import numpy as np
import pandas as pd


def _plot_single_shaded_line(axis, xx, y_lower, y_middle, y_upper, **kwargs):
    """Plot a line with shaded area between y_lower and y_upper bounds."""
    assert isinstance(
        axis, matplotlib.axes._axes.Axes
    ), "First argument must be a matplotlib axis"
    assert (
        len(xx) == len(y_middle) == len(y_lower) == len(y_upper)
    ), "All arrays must have the same length"
    alpha = kwargs.pop("alpha", 0.3)
    color = kwargs.get("color", axis._get_lines.get_next_color())
    axis.plot(xx, y_middle, color=color, **kwargs)
    axis.fill_between(xx, y_lower, y_upper, alpha=alpha, color=color)
    return axis, pd.DataFrame(
        {"x": xx, "y_lower": y_lower, "y_middle": y_middle, "y_upper": y_upper}
    )


def _plot_shaded_lines(
    axis, xs, ys_lower, ys_middle, ys_upper, colors=None, **kwargs
):
    """Plot multiple lines with shaded areas between ys_lower and ys_upper bounds."""
    assert isinstance(
        axis, matplotlib.axes._axes.Axes
    ), "First argument must be a matplotlib axis"
    results = []
    if colors is not None:
        if not isinstance(colors, list):
            colors = [colors] * len(xs)
        for idx, (xx, y_lower, y_middle, y_upper) in enumerate(
            zip(xs, ys_lower, ys_middle, ys_upper)
        ):
            this_kwargs = kwargs.copy()
            this_kwargs["color"] = colors[idx]
            _, result_df = _plot_single_shaded_line(
                axis, xx, y_lower, y_middle, y_upper, **this_kwargs
            )
            results.append(result_df)
    else:
        for xx, y_lower, y_middle, y_upper in zip(
            xs, ys_lower, ys_middle, ys_upper
        ):
            _, result_df = _plot_single_shaded_line(
                axis, xx, y_lower, y_middle, y_upper, **kwargs
            )
            results.append(result_df)
    return axis, results


def plot_shaded_line(
    axis, xs, ys_lower, ys_middle, ys_upper, colors=None, **kwargs
):
    """
    Plot a line with shaded area, automatically switching between single and multiple line versions.

    Args:
        axis: matplotlib axis
        xs: x values (single array or list of arrays)
        ys_lower: lower bound y values (single array or list of arrays)
        ys_middle: middle y values (single array or list of arrays)
        ys_upper: upper bound y values (single array or list of arrays)
        colors: color or list of colors for the lines
        **kwargs: additional plotting parameters

    Returns:
        tuple: (axis, DataFrame or list of DataFrames with plot data)
    """
    # Check if inputs are lists of arrays
    if isinstance(xs, list) and all(isinstance(x, np.ndarray) for x in xs):
        return _plot_shaded_lines(
            axis, xs, ys_lower, ys_middle, ys_upper, colors=colors, **kwargs
        )
    else:
        if colors is not None:
            kwargs["color"] = colors[0] if isinstance(colors, list) else colors
        return _plot_single_shaded_line(
            axis, xs, ys_lower, ys_middle, ys_upper, **kwargs
        )

# EOF