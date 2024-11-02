# src from here --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-07-05 20:06:17 (ywatanabe)"
# # /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_ecdf.py
# 
# import mngs
# import numpy as np
# import pandas as pd
# 
# # def ecdf(ax, data):
# #     data = np.hstack(data)
# #     data = data[~np.isnan(data)]
# #     nn = len(data)
# 
# #     data_sorted = np.sort(data)
# #     ecdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
# 
# #     ax.step(data_sorted, ecdf, where="post")
# #     ax.plot(data_sorted, ecdf, marker=".", linestyle="none")
# 
# #     df = pd.DataFrame(
# #         {
# #             "x": data_sorted,
# #             "y": ecdf,
# #             "n": nn,
# #         }
# #     )
# #     return ax, df
# 
# 
# def ecdf(ax, data, **kwargs):
#     # Flatten and remove NaN values
#     data = np.hstack(data)
#     data = data[~np.isnan(data)]
#     nn = len(data)
# 
#     # Sort the data and compute the ECDF values
#     data_sorted = np.sort(data)
#     ecdf_perc = 100 * np.arange(1, len(data_sorted) + 1) / len(data_sorted)
# 
#     # Create the pseudo x-axis for step plotting
#     x_step = np.repeat(data_sorted, 2)[1:]
#     y_step = np.repeat(ecdf_perc, 2)[:-1]
# 
#     # Plot the ECDF_PERC using steps
#     ax.plot(x_step, y_step, drawstyle="steps-post", **kwargs)
# 
#     # Scatter the original data points
#     ax.plot(data_sorted, ecdf_perc, marker=".", linestyle="none")
# 
#     # Set ylim, xlim, and aspect ratio
#     ax.set_ylim(0, 100)  # Set y-axis limits
#     ax.set_xlim(0, 1.0)  # Set x-axis limits
#     # ax.set_aspect(1.0)  # Set aspect ratio
# 
#     # Create a DataFrame to hold the ECDF_PERC data
#     df = mngs.pd.force_df(
#         {
#             "x": data_sorted,
#             "y": ecdf_perc,
#             "n": nn,
#             "x_step": x_step,
#             "y_step": y_step,
#         }
#     )
# 
#     # Return the matplotlib axis and DataFrame
#     return ax, df
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
    sys.path.insert(0, project_root)

from src.mngs.plt/ax/_ecdf.py import *

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
