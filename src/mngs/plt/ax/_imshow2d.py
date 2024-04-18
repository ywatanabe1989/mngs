#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-16 20:47:36"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script does XYZ.
"""

import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np


# Functions
def imshow2d(
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

    assert arr_2d.ndim == 2

    # Transposes arr_2d for correct orientation
    arr_2d = arr_2d.T

    # Cals the original ax.imshow() method on the transposed array
    im = ax.imshow(
        arr_2d, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, **kwargs
    )

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

    # # Set ticks for x and y axes
    # xticks = np.arange(arr_2d.shape[1])  # Correct after transposition
    # ax.set_xticks(xticks)
    # ax.set_xticklabels([str(xt) for xt in xticks])

    # yticks = np.arange(arr_2d.shape[0])  # Correct after transposition
    # ax.set_yticks(yticks)
    # ax.set_yticklabels([str(yt) for yt in yticks])

    return ax


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    # Example usage
    fig, ax = plt.subplots()
    data = np.random.rand(10, 20)  # Random data
    imshow2d(ax, data)
    plt.show()

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
/ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/plt/ax/_imshow2d.py
"""
