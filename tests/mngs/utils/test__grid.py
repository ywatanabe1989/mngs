# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/utils/_grid.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-22 23:54:02"
# # Author: Yusuke Watanabe (ywata1989@gmail.com)
# 
# """
# This script defines mngs.ml.utils.grid_search
# """
# 
# # Imports
# import itertools as _itertools
# import random as _random
# import sys as _sys
# 
# import matplotlib.pyplot as _plt
# import mngs as _mngs
# 
# 
# # Functions
# def yield_grids(params_grid: dict, random=False):
#     """
#     Generator function that yields combinations of parameters from a grid.
# 
#     Args:
#         params_grid (dict): A dictionary where keys are parameter names and values are lists of parameter values.
#         random (bool): If True, yields the parameter combinations in random order.
# 
#     Yields:
#         dict: A dictionary of parameters for one set of conditions from the grid.
# 
#     Example:
#         # Parameters
#         params_grid = {
#             "batch_size": [2**i for i in range(7)],
#             "n_chs": [2**i for i in range(7)],
#             "seq_len": [2**i for i in range(15)],
#             "fs": [2**i for i in range(8, 11)],
#             "n_segments": [2**i for i in range(6)],
#             "n_bands_pha": [2**i for i in range(7)],
#             "n_bands_amp": [2**i for i in range(7)],
#             "precision": ['fp16', 'fp32'],
#             "device": ['cpu', 'cuda'],
#             "package": ['tensorpac', 'mngs'],
#         }
# 
#         # Example of using the generator
#         for param_dict in yield_grids(params_grid, random=True):
#             print(param_dict)
#     """
#     combinations = list(_itertools.product(*params_grid.values()))
#     if random:
#         _random.shuffle(combinations)  # [REVISED]
#     for values in combinations:
#         yield dict(zip(params_grid.keys(), values))
# 
# 
# # def yield_grids(params_grid: dict, random=False):
# #     """
# #     Generator function that yields combinations of parameters from a grid.
# 
# #     Args:
# #         params_grid (dict): A dictionary where keys are parameter names and values are lists of parameter values.
# 
# #     Yields:
# #         dict: A dictionary of parameters for one set of conditions from the grid.
# 
# #     Example:
# #         # Parameters
# #         params_grid = {
# #             "batch_size": [2**i for i in range(7)],
# #             "n_chs": [2**i for i in range(7)],
# #             "seq_len": [2**i for i in range(15)],
# #             "fs": [2**i for i in range(8, 11)],
# #             "n_segments": [2**i for i in range(6)],
# #             "n_bands_pha": [2**i for i in range(7)],
# #             "n_bands_amp": [2**i for i in range(7)],
# #             "precision": ['fp16', 'fp32'],
# #             "device": ['cpu', 'cuda'],
# #             "package": ['tensorpac', 'mngs'],
# #         }
# 
# #         # Example of using the generator
# #         for param_dict in yield_grids(params_grid):
# #             print(param_dict)
# #     """
# #     print(f"\nThe Number of Combinations: {count_grids(params_grid):,}")
# 
# #     for values in _itertools.product(*params_grid.values()):
# #         yield dict(zip(params_grid.keys(), values))
# 
# 
# def count_grids(params_grid):
#     """
#     Calculate the total number of combinations possible from the given parameter grid.
# 
#     Args:
#         params_grid (dict): A dictionary where keys are parameter names and values are lists of parameter values.
# 
#     Returns:
#         int: The total number of combinations that can be generated from the parameter grid.
#     """
#     # Get the number of values for each parameter and multiply them
#     num_combinations = 1
#     for values in params_grid.values():
#         num_combinations *= len(values)
#     return num_combinations
# 
# 
# if __name__ == "__main__":
#     import pandas as pd
# 
#     # Start
#     CONFIG, _sys.stdout, _sys.stderr, _plt, CC = _mngs.gen.start(
#         _sys, _plt, verbose=False
#     )
# 
#     # Parameters
#     N = 15
#     print(pd.DataFrame(pd.Series({f"2^{ii}": 2**ii for ii in range(N)})))
# 
#     params_grid = {
#         "batch_size": [2**i for i in [3, 4, 5, 6]],
#         "n_chs": [2**i for i in [3, 4, 5, 6]],
#         "seq_len": [2**i for i in range(8, 13)],
#         "fs": [2**i for i in range(7, 10)],
#         "n_segments": [2**i for i in range(5)],
#         "n_bands_pha": [2**i for i in range(7)],
#         "n_bands_amp": [2**i for i in range(7)],
#         "precision": ["fp16", "fp32"],
#         "device": ["cpu", "cuda"],
#         "package": ["tensorpac", "_mngs"],
#     }
# 
#     print(params_grid)
#     print(f"{count_grids(params_grid):,}")
# 
#     # Example of using the generator
#     for param_dict in yield_grids(params_grid):
#         print(param_dict)
# 
#     # Close
#     _mngs.gen.close(CONFIG, verbose=False, notify=False)
# 
# # EOF
# 
# """
# /home/ywatanabe/proj/entrance/_mngs/ml/utils/grid_search.py
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

from mngs.utils._grid import *

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
