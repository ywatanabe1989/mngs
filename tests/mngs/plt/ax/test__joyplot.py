# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-06 00:04:26 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/plt/ax/_joyplot.py
# 
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-08-26 08:38:55 (ywatanabe)"
# # /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_joyplot.py
# 
# import joypy
# import mngs
# from ._set_xyt import set_xyt
# 
# # def plot_joy(df, cols_NT):
# #     df_plot = df[cols_NT]
# #     fig, axes = joypy.joyplot(
# #         data=df_plot,
# #         colormap=plt.cm.viridis,
# #         title="Distribution of Ranked Data",
# #         labels=cols_NT,
# #         overlap=0.5,
# #         orientation="vertical",
# #     )
# #     plt.xlabel("Variables")
# #     plt.ylabel("Rank")
# #     return fig
# 
# 
# def joyplot(ax, data, **kwargs):
#     fig, axes = joypy.joyplot(
#         data=data,
#         **kwargs,
#     )
# 
#     if kwargs.get("orientation") == "vertical":
#         xlabel = None
#         ylabel = "Density"
#     else:
#         xlabel = "Density"
#         ylabel = None
# 
#     ax = set_xyt(ax, xlabel, ylabel, "Joyplot")
# 
#     return ax
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

from ...src.mngs..plt.ax._joyplot import *

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
