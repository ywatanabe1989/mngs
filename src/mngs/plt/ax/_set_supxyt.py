#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-17 08:37:31"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script does XYZ.
"""

# Imports
import matplotlib.pyplot as plt


# Functions
def set_supxyt(ax, xlabel=None, ylabel=None, title=None):
    fig = ax.get_figure()
    """Sets xlabel, ylabel and title"""
    if xlabel is not None:
        fig.supxlabel(xlabel)
    if ylabel is not None:
        fig.supylabel(ylabel)
    if title is not None:
        fig.suptitle(title)
    return ax


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    # (YOUR AWESOME CODE)

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
/ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/plt/ax/_set_lt.py
"""
