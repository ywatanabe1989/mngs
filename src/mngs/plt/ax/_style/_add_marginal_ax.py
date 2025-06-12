#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 20:18:52 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_add_marginal_ax.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/plt/ax/_add_marginal_ax.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ....plt.utils import assert_valid_axis


def add_marginal_ax(axis, place, size=0.2, pad=0.1):
<<<<<<< HEAD
    """Add a marginal axis to an existing axis.

    Creates a new axis adjacent to an existing one, useful for adding
    marginal distributions, colorbars, or supplementary plots. The new
    axis shares one dimension with the original axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The main axis to which a marginal axis will be added.
    place : {'left', 'right', 'top', 'bottom'}
        Position where the marginal axis should be placed.
    size : float, optional
        Size of the marginal axis as a fraction of the main axis.
        Default is 0.2 (20%). For left/right placement, this represents
        the width; for top/bottom, the height.
    pad : float, optional
        Padding between the main and marginal axes. Default is 0.1.

    Returns
    -------
    matplotlib.axes.Axes
        The newly created marginal axis.

    Raises
    ------
    AssertionError
        If axis is not a matplotlib axes object.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.scatter(x, y)
    >>> ax_top = add_marginal_ax(ax, 'top', size=0.15)
    >>> ax_top.hist(x, bins=30)  # Add x-distribution

    >>> # Add marginal distributions to scatter plot
    >>> ax_right = add_marginal_ax(ax, 'right', size=0.2)
    >>> ax_right.hist(y, bins=30, orientation='horizontal')

    See Also
    --------
    matplotlib.axes_grid1.make_axes_locatable : Used internally

    Notes
    -----
    The marginal axis will automatically align with the main axis
    and adjust when the main axis is resized.
    """
    assert isinstance(
        axis, matplotlib.axes._axes.Axes
    ), "First argument must be a matplotlib axis"
=======
    """
    Add a marginal axis to the specified side of an existing axis.
    
    Arguments:
        axis (matplotlib.axes.Axes or mngs.plt._subplots.AxisWrapper): The axis to which a marginal axis will be added.
        place (str): Where to place the marginal axis ('top', 'right', 'bottom', or 'left').
        size (float, optional): Fractional size of the marginal axis relative to the main axis. Defaults to 0.2.
        pad (float, optional): Padding between the axes. Defaults to 0.1.
        
    Returns:
        matplotlib.axes.Axes: The newly created marginal axis.
    """
    assert_valid_axis(axis, "First argument must be a matplotlib axis or mngs axis wrapper")
>>>>>>> origin/main

    divider = make_axes_locatable(axis)

    size_perc_str = f"{size*100}%"
    if place in ["left", "right"]:
        size = 1.0 / size

    axis_marginal = divider.append_axes(place, size=size_perc_str, pad=pad)
    axis_marginal.set_box_aspect(size)

    return axis_marginal


# EOF
