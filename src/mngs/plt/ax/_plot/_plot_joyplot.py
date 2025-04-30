#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 08:51:52 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_plot/_plot_joyplot.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/ax/_plot/_plot_joyplot.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import warnings

import joypy

from .._adjust._set_xyt import set_xyt as mngs_plt_set_xyt


def _plot_joyplot(ax, data, orientation, **kwargs):
    # FIXME; orientation should be handled
    fig, axes = joypy.joyplot(
        data=data,
        **kwargs,
    )

    if orientation == "vertical":
        ax = mngs_plt_set_xyt(ax, None, "Density", "Joyplot")
    elif orientation == "horizontal":
        ax = mngs_plt_set_xyt(ax, "Density", None, "Joyplot")
    else:
        warnings.warn(
            "orientation must be either of 'vertical' or 'horizontal'"
        )

    return ax


def plot_vertical_joyplot(ax, data, **kwargs):
    return _plot_joyplot(ax, data, "vertical", **kwargs)


def plot_horizontal_joyplot(ax, data, **kwargs):
    return _plot_joyplot(ax, data, "horizontal", **kwargs)

# EOF