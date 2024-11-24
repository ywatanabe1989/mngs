# src from here --------------------------------------------------------------------------------
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

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs..plt._interp_colors import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        pass

    def test_edge_cases(self):
        # Edge case testing
        pass

    def test_error_handling(self):
        # Error handling testing
        pass
