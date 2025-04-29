#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:59:13 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__interp_colors.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__interp_colors.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib.colors as mcolors
import numpy as np


def test_interp_colors():
    from mngs.plt._interp_colors import interp_colors

    # Test with basic colors and small number of points
    color_start = "red"
    color_end = "blue"
    num_points = 5

    colors = interp_colors(color_start, color_end, num_points)

    # Check that we get the expected number of colors
    assert len(colors) == num_points

    # Check that each color is a list of 4 values (RGBA)
    for color in colors:
        assert len(color) == 4
        for value in color:
            assert 0 <= value <= 1

    # Check that the first and last colors match the inputs
    start_rgba = np.array(mcolors.to_rgba(color_start)).round(3)
    end_rgba = np.array(mcolors.to_rgba(color_end)).round(3)

    np.testing.assert_almost_equal(colors[0], start_rgba, decimal=3)
    np.testing.assert_almost_equal(colors[-1], end_rgba, decimal=3)


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_interp_colors.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-06 10:01:32 (ywatanabe)"
# # /home/ywatanabe/proj/mngs/src/mngs/plt/_interp_colors.py
# 
# 
# import matplotlib.colors as mcolors
# import numpy as np
# 
# 
# def interp_colors(color_start, color_end, num_points, round=3):
#     # Convert colors to RGBA if they are not already in that format
#     color_start_rgba = np.array(mcolors.to_rgba(color_start))
#     color_end_rgba = np.array(mcolors.to_rgba(color_end))
# 
#     # Generate a sequence of RGBA values
#     # np.linspace works on a per-component basis when the inputs are arrays
#     rgba_values = np.linspace(
#         color_start_rgba, color_end_rgba, num_points
#     ).round(round)
# 
#     # Return the list of RGBA tuples
#     return [list(color) for color in rgba_values]
# 
#     # # Generate a linear interpolation
#     # return [for color in np.linspace(color_start_rgba, color_end_rgba, num_points)]
#     # # return [
#     # #     mcolors.to_hex(color)
#     # #     for color in np.linspace(color_start_rgba, color_end_rgba, num_points)
#     # # ]

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_interp_colors.py
# --------------------------------------------------------------------------------
