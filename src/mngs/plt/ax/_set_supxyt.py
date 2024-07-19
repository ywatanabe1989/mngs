#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-13 07:56:46 (ywatanabe)"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script does XYZ.
"""

# Imports
import matplotlib.pyplot as plt


# Functions
def set_supxyt(ax, xlabel=False, ylabel=False, title=False):
    """Sets xlabel, ylabel and title"""
    fig = ax.get_figure()

    if xlabel is not False:
        fig.supxlabel(xlabel)

    if ylabel is not False:
        fig.supylabel(ylabel)

    if title is not False:
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
