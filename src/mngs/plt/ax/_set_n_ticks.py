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

    if n_xticks is not None:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(n_xticks))

    if n_yticks is not None:
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(n_yticks))

    # Force the figure to redraw to reflect changes
    ax.figure.canvas.draw()

    return ax
