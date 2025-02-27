# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "ywatanabe (2024-11-03 00:48:22)"
# # File: ./mngs_repo/src/mngs/dict/_to_str.py
# 
# 
# def to_str(dictionary, delimiter="_"):
#     """
#     Convert a dictionary to a string representation.
# 
#     Example
#     -------
#     input_dict = {'a': 1, 'b': 2, 'c': 3}
#     result = dict2str(input_dict)
#     print(result)  # Output: a-1_b-2_c-3
# 
#     Parameters
#     ----------
#     dictionary : dict
#         The input dictionary to be converted.
#     delimiter : str, optional
#         The separator between key-value pairs (default is "_").
# 
#     Returns
#     -------
#     str
#         A string representation of the input dictionary.
#     """
#     return delimiter.join(
#         f"{key}-{value}" for key, value in dictionary.items()
#     )
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

from mngs..dict._to_str import *

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
