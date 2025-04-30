#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 21:18:12 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_interp_colors.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_interp_colors.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib.colors as mcolors
import numpy as np
from mngs.decorators import deprecated


def gen_interp_colors(color_start, color_end, num_points, round=3):
    color_start_rgba = np.array(mcolors.to_rgba(color_start))
    color_end_rgba = np.array(mcolors.to_rgba(color_end))
    rgba_values = np.linspace(
        color_start_rgba, color_end_rgba, num_points
    ).round(round)
    return [list(color) for color in rgba_values]


@deprecated("Use gen_interp_colors instead")
def interp_colors(color_start, color_end, num_points, round=3):
    return gen_interp_colors(color_start, color_end, num_points, round=round)

# EOF