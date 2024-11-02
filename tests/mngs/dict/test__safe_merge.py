# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "ywatanabe (2024-11-03 00:47:50)"
# # File: ./mngs_repo/src/mngs/dict/_safe_merge.py
# 
# """
# Functionality:
#     - Safely merges multiple dictionaries without overlapping keys
# Input:
#     - Multiple dictionaries to be merged
# Output:
#     - A single merged dictionary
# Prerequisites:
#     - mngs.gen package with search function
# """
# 
# from typing import Any as _Any
# from typing import Dict
# 
# from ..utils import search
# 
# 
# def safe_merge(*dicts: Dict[_Any, _Any]) -> Dict[_Any, _Any]:
#     """Merges dictionaries while checking for key conflicts.
# 
#     Example
#     -------
#     >>> dict1 = {'a': 1, 'b': 2}
#     >>> dict2 = {'c': 3, 'd': 4}
#     >>> safe_merge(dict1, dict2)
#     {'a': 1, 'b': 2, 'c': 3, 'd': 4}
# 
#     Parameters
#     ----------
#     *dicts : Dict[_Any, _Any]
#         Variable number of dictionaries to merge
# 
#     Returns
#     -------
#     Dict[_Any, _Any]
#         Merged dictionary
# 
#     Raises
#     ------
#     ValueError
#         If overlapping keys are found between dictionaries
#     """
#     try:
#         merged_dict: Dict[_Any, _Any] = {}
#         for current_dict in dicts:
#             overlap_check = search(
#                 merged_dict.keys(), current_dict.keys(), only_perfect_match=True
#             )
#             if overlap_check != ([], []):
#                 raise ValueError("Overlapping keys found between dictionaries")
#             merged_dict.update(current_dict)
#         return merged_dict
#     except Exception as error:
#         raise ValueError(f"Dictionary merge failed: {str(error)}")
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

from src.mngs.dict/_safe_merge.py import *

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
