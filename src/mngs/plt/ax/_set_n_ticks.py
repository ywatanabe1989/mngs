#!/usr/bin/env python


import matplotlib


def set_n_ticks(
    ax,
    n_xticks=4,
    n_yticks=4,
):
    """
    Example:
        ax = set_n_ticks(ax)
    """

    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(n_xticks))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(n_yticks))

    return ax
