# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_imshow2d.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-09-28 10:19:09 (ywatanabe)"
# # Author: Yusuke Watanabe (ywata1989@gmail.com)
# 
# """This script does XYZ."""
# 
# import sys
# 
# import matplotlib.pyplot as plt
# import mngs
# import numpy as np
# 
# 
# # Functions
# def imshow2d(
#     ax,
#     arr_2d,
#     cbar=True,
#     cbar_label=None,
#     cbar_shrink=1.0,
#     cbar_fraction=0.046,
#     cbar_pad=0.04,
#     cmap="viridis",
#     aspect="auto",
#     vmin=None,
#     vmax=None,
#     **kwargs,
# ):
#     """
#     Imshows an two-dimensional array with theese two conditions:
#     1) The first dimension represents the x dim, from left to right.
#     2) The second dimension represents the y dim, from bottom to top
#     """
# 
#     assert arr_2d.ndim == 2
# 
#     # Transposes arr_2d for correct orientation
#     arr_2d = arr_2d.T
# 
#     # Cals the original ax.imshow() method on the transposed array
#     im = ax.imshow(
#         arr_2d, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, **kwargs
#     )
# 
#     # Color bar
#     if cbar:
#         fig = ax.get_figure()
#         _cbar = fig.colorbar(
#             im, ax=ax, shrink=cbar_shrink, fraction=cbar_fraction, pad=cbar_pad
#         )
#         if cbar_label:
#             _cbar.set_label(cbar_label)
# 
#     # Invert y-axis to match typical image orientation
#     ax.invert_yaxis()
# 
#     return ax
# 
# 
# if __name__ == "__main__":
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
# 
#     # Example usage
#     fig, ax = plt.subplots()
#     data = np.random.rand(10, 20)  # Random data
#     imshow2d(ax, data)
#     plt.show()
# 
#     # Close
#     mngs.gen.close(CONFIG)
# 
# # EOF
# 
# """
# /ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/plt/ax/_imshow2d.py
# """

#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.plt.ax._imshow2d import *

class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        raise NotImplementedError("Test not yet implemented")

    def test_edge_cases(self):
        # Edge case testing
        raise NotImplementedError("Test not yet implemented")

    def test_error_handling(self):
        # Error handling testing
        raise NotImplementedError("Test not yet implemented")

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
