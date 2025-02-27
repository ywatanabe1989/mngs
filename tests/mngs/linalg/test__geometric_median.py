# src from here --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-09-26 19:41:01 (ywatanabe)"
# # /home/ywatanabe/proj/mngs/src/mngs/linalg/_geometric_median.py
# 
# 
# """
# This script does XYZ.
# """
# 
# import torch
# from geom_median.torch import compute_geometric_median
# from ..decorators import torch_fn
# 
# # @torch_fn
# # def geometric_median(xx, dim=-1):
# #     indi = [slice(None) for _ in range(xx.ndim)]
# #     indi[dim] = slice(None)
# #     xx[indi]  # how can I loop over the designated dim??
# 
# #     return compute_geometric_median(xx).median
# 
# 
# @torch_fn
# def geometric_median(xx, dim=-1):
# 
#     # Ensure dim is a positive index
#     if dim < 0:
#         dim = xx.ndim + dim
#         dim = torch.tensor(dim).to(xx.device)
# 
#     # Create a list of slices to access all elements along each dimension
#     indi = [slice(None)] * xx.ndim
# 
#     # Get the size of the dimension we want to loop over
#     dim_size = xx.shape[dim]
# 
#     points = []
#     # Loop over each index in the specified dimension
#     for i in range(dim_size):
#         indi[
#             dim
#         ] = i
#         # Set the slice for the current index in the target dimension
#         slice_data = xx[tuple(indi)]  # Extract the data for the current index
#         points.append(slice_data)
# 
#     out = compute_geometric_median(points).median
# 
#     return out
# 
# 
# if __name__ == "__main__":
#     # # Argument Parser
#     # import argparse
#     # parser = argparse.ArgumentParser(description='')
#     # parser.add_argument('--var', '-v', type=int, default=1, help='')
#     # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
#     # args = parser.parse_args()
# 
#     # Main
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False
#     )
#     main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)
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

from ...src.mngs..linalg._geometric_median import *

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
