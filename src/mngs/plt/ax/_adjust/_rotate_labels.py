#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-27 13:24:32 (ywatanabe)"
# /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/plt/ax/_rotate_labels.py

"""This script does XYZ."""

"""Imports"""
def rotate_labels(ax, x=45, y=45, x_ha='center', y_ha='center'):
    """
    Rotate x and y axis labels of a matplotlib Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to modify.
    x : float, optional
        Rotation angle for x-axis labels in degrees. Default is 45.
    y : float, optional
        Rotation angle for y-axis labels in degrees. Default is 45.
    x_ha : str, optional
        Horizontal alignment for x-axis labels. Default is 'center'.
    y_ha : str, optional
        Horizontal alignment for y-axis labels. Default is 'center'.

    Returns
    -------
    matplotlib.axes.Axes
        The modified Axes object.

    Example
    -------
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    rotate_labels(ax)
    plt.show()
    """
    # Get current tick positions
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    # Set ticks explicitly
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Set labels with rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=x, ha=x_ha)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=y, ha=y_ha)
    return ax
