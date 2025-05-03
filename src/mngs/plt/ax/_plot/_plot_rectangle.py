#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 08:45:44 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_plot/_plot_rectangle.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/ax/_plot/_plot_rectangle.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from matplotlib.patches import Rectangle


def plot_rectangle(ax, xx, yy, ww, hh, **kwargs):
    ax.add_patch(Rectangle((xx, yy), ww, hh, **kwargs))
    return ax

# EOF