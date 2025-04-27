# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_ecdf.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-06 00:03:51 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/plt/ax/_ecdf.py
# 
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-07-05 20:06:17 (ywatanabe)"
# # ./src/mngs/plt/ax/_ecdf.py
# 
# import mngs
# import numpy as np
# import pandas as pd
# from ...pd._force_df import force_df
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
#     df = force_df(
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

from mngs.plt.ax._ecdf import *

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
