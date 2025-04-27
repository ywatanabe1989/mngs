#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 16:38:41 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__interp_colors.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__interp_colors.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/_interp_colors.py
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

import sys

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.plt._interp_colors import interp_colors


def test_interp_colors_length_and_type():
    result = interp_colors("red", "blue", 5)
    assert isinstance(result, list)
    assert len(result) == 5
    for rgba in result:
        assert isinstance(rgba, list)
        assert len(rgba) == 4
        assert all(isinstance(c, float) for c in rgba)


def test_interp_colors_endpoints():
    result = interp_colors("red", "red", 3)
    assert result[0] == result[-1]

# EOF