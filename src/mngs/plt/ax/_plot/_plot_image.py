#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 08:39:46 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_plot/_plot_image2d.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/plt/ax/_plot/_plot_image2d.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib


def plot_image(
    ax,
    arr_2d,
    cbar=True,
    cbar_label=None,
    cbar_shrink=1.0,
    cbar_fraction=0.046,
    cbar_pad=0.04,
    cmap="viridis",
    aspect="auto",
    vmin=None,
    vmax=None,
    **kwargs,
):
    """
    Imshows an two-dimensional array with theese two conditions:
    1) The first dimension represents the x dim, from left to right.
    2) The second dimension represents the y dim, from bottom to top
    """
    assert isinstance(ax, matplotlib.axes._axes.Axes)
    assert arr_2d.ndim == 2

    if kwargs.get("xyz"):
        kwargs.pop("xyz")

    # Transposes arr_2d for correct orientation
    arr_2d = arr_2d.T

    # Cals the original ax.imshow() method on the transposed array
    im = ax.imshow(arr_2d, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, **kwargs)

    # Color bar
    if cbar:
        fig = ax.get_figure()
        _cbar = fig.colorbar(
            im, ax=ax, shrink=cbar_shrink, fraction=cbar_fraction, pad=cbar_pad
        )
        if cbar_label:
            _cbar.set_label(cbar_label)

    # Invert y-axis to match typical image orientation
    ax.invert_yaxis()

    return ax


# EOF
