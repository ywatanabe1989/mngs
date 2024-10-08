#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-13 08:14:19 (ywatanabe)"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script does XYZ.
"""

# Imports
import matplotlib.pyplot as plt


# Functions
def set_xyt(ax, x=False, y=False, t=False):
    """Sets xlabel, ylabel and title"""

    if x is not False:
        ax.set_xlabel(x)

    if y is not False:
        ax.set_ylabel(y)

    if t is not False:
        ax.set_title(t)

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
