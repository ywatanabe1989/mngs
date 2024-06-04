#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-16 21:26:05"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script does XYZ.
"""

# Imports
import matplotlib.pyplot as plt


# Functions
def set_xyt(ax, xlabel=None, ylabel=None, title=None):
    """Sets xlabel, ylabel and title"""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
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
