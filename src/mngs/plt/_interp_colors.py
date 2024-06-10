#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-06 10:01:32 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/_interp_colors.py


import matplotlib.colors as mcolors
import numpy as np


def interp_colors(color_start, color_end, num_points, round=3):
    # Convert colors to RGBA if they are not already in that format
    color_start_rgba = np.array(mcolors.to_rgba(color_start))
    color_end_rgba = np.array(mcolors.to_rgba(color_end))

    # Generate a sequence of RGBA values
    # np.linspace works on a per-component basis when the inputs are arrays
    rgba_values = np.linspace(
        color_start_rgba, color_end_rgba, num_points
    ).round(round)

    # Return the list of RGBA tuples
    return [list(color) for color in rgba_values]

    # # Generate a linear interpolation
    # return [for color in np.linspace(color_start_rgba, color_end_rgba, num_points)]
    # # return [
    # #     mcolors.to_hex(color)
    # #     for color in np.linspace(color_start_rgba, color_end_rgba, num_points)
    # # ]
