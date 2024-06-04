#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-15 18:55:49"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script does XYZ.
"""

import sys

import matplotlib.pyplot as plt

# Imports
import mngs
from matplotlib.patches import Rectangle


# Functions
def rectangle(ax, xx, yy, ww, hh, **kwargs):
    ax.add_patch(Rectangle((xx, yy), ww, hh, **kwargs))
    return ax


# (YOUR AWESOME CODE)

if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    # (YOUR AWESOME CODE)

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
/ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/plt/ax/_rectangle.py
"""
