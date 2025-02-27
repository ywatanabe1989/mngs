# src from here --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-07-05 06:01:32 (ywatanabe)"
# # /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_fill_between_v.py
# 
# 
# """
# This script does XYZ.
# """
# 
# 
# """
# Imports
# """
# 
# import numpy as np
# 
# 
# def fillv(axes, starts, ends, color="red", alpha=0.2):
#     """
#     Fill between specified start and end intervals on an axis or array of axes.
# 
#     Parameters
#     ----------
#     axes : matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes
#         The axis object(s) to fill intervals on.
#     starts : array-like
#         Array-like of start positions.
#     ends : array-like
#         Array-like of end positions.
#     color : str, optional
#         The color to use for the filled regions. Default is "red".
#     alpha : float, optional
#         The alpha blending value, between 0 (transparent) and 1 (opaque). Default is 0.2.
# 
#     Returns
#     -------
#     list
#         List of axes with filled intervals.
#     """
# 
#     is_axes = isinstance(axes, np.ndarray)
# 
#     axes = axes if isinstance(axes, np.ndarray) else [axes]
# 
#     for ax in axes:
#         for start, end in zip(starts, ends):
#             ax.axvspan(start, end, color=color, alpha=alpha)
# 
#     if not is_axes:
#         return axes[0]
#     else:
#         return axes
# 
# 
# # EOF

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

from mngs..plt.ax._fillv import *

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
