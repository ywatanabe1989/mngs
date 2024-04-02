#!/usr/bin/env python


import matplotlib


def set_n_ticks(
    ax,
    n_xticks=None,
    n_yticks=None,
):
    """
    Example:
        ax = set_n_ticks(ax)
    """

    if n_xticks is not None:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(n_xticks))

    if n_yticks is not None:
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(n_yticks))

    return ax
