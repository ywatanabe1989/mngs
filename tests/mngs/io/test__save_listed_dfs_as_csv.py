# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 21:28:56 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_save_listed_dfs_as_csv.py
# 
# import csv
# import numpy as np
# from ._mv_to_tmp import _mv_to_tmp
# 
# def _save_listed_dfs_as_csv(
#     listed_dfs,
#     spath_csv,
#     indi_suffix=None,
#     overwrite=False,
#     verbose=False,
# ):
#     """listed_dfs:
#         [df1, df2, df3, ..., dfN]. They will be written vertically in the order.
# 
#     spath_csv:
#         /hoge/fuga/foo.csv
# 
#     indi_suffix:
#         At the left top cell on the output csv file, '{}'.format(indi_suffix[i])
#         will be added, where i is the index of the df.On the other hand,
#         when indi_suffix=None is passed, only '{}'.format(i) will be added.
#     """
# 
#     if overwrite == True:
#         _mv_to_tmp(spath_csv, L=2)
# 
#     indi_suffix = (
#         np.arange(len(listed_dfs)) if indi_suffix is None else indi_suffix
#     )
#     for i, df in enumerate(listed_dfs):
#         with open(spath_csv, mode="a") as f:
#             f_writer = csv.writer(f)
#             i_suffix = indi_suffix[i]
#             f_writer.writerow(["{}".format(indi_suffix[i])])
#         df.to_csv(spath_csv, mode="a", index=True, header=True)
#         with open(spath_csv, mode="a") as f:
#             f_writer = csv.writer(f)
#             f_writer.writerow([""])
#     if verbose:
#         print("Saved to: {}".format(spath_csv))
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

from src.mngs.io/_save_listed_dfs_as_csv.py import *

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
