#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-07 23:44:26 (ywatanabe)"


def hide_spines(ax, tgts=["top", "right", "bottom", "left"]):
    """
    Hides specified spines from a matplotlib Axes.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to modify.
    - tgts (list of str): List of spines to hide from the Axes.

    Returns:
    - ax (matplotlib.axes.Axes): The modified Axes object with spines removed.
    """
    for tgt in tgts:
        ax.spines[tgt].set_visible(False)

    return ax
