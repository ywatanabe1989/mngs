#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-15 18:46:38"


def hide_spines(
    ax,
    tgts=["top", "right", "bottom", "left"],
    hide_ticks=False,
    hide_labels=False,
):
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
        if hide_ticks:
            if tgt in ["top", "bottom"]:
                ax.xaxis.set_ticks_position("none")
            elif tgt in ["left", "right"]:
                ax.yaxis.set_ticks_position("none")
        if hide_labels:
            if tgt in ["top", "bottom"]:
                ax.set_xticklabels([])
            elif tgt in ["left", "right"]:
                ax.set_yticklabels([])

    return ax
