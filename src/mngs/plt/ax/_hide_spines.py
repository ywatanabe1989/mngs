#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-26 20:03:45 (ywatanabe)"


def hide_spines(
    ax,
    top=True,
    bottom=True,
    left=True,
    right=True,
    ticks=True,
    labels=True,
):
    """
    Hides the specified spines of a matplotlib Axes object and optionally removes the ticks and labels.

    This function is designed to work with matplotlib Axes objects. It allows for a cleaner, more minimalist
    presentation of plots by hiding the spines (the lines denoting the boundaries of the plot area) and optionally
    removing the ticks and labels from the axes.

    Arguments:
        ax (matplotlib.axes.Axes): The Axes object for which the spines will be hidden.
        top (bool, optional): If True, hides the top spine. Defaults to True.
        bottom (bool, optional): If True, hides the bottom spine. Defaults to True.
        left (bool, optional): If True, hides the left spine. Defaults to True.
        right (bool, optional): If True, hides the right spine. Defaults to True.
        ticks (bool, optional): If True, removes the ticks from the hidden spines' axes. Defaults to True.
        labels (bool, optional): If True, removes the labels from the hidden spines' axes. Defaults to True.

    Returns:
        matplotlib.axes.Axes: The modified Axes object with the specified spines hidden.

    Example:
        >>> fig, ax = plt.subplots()
        >>> hide_spines(ax, top=False, labels=False)
        >>> plt.show()
    """

    tgts = []
    if top:
        tgts.append("top")
    if bottom:
        tgts.append("bottom")
    if left:
        tgts.append("left")
    if right:
        tgts.append("right")

    for tgt in tgts:
        # Spines
        ax.spines[tgt].set_visible(False)

        # Ticks
        if ticks:
            if tgt == "bottom":
                ax.xaxis.set_ticks_position("none")
            elif tgt == "left":
                ax.yaxis.set_ticks_position("none")

        # Labels
        if labels:
            if tgt == "bottom":
                ax.set_xticklabels([])
            elif tgt == "left":
                ax.set_yticklabels([])

    return ax
