# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 04:05:41 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/str/_grep.py
# 
# import re
# 
# def grep(str_list, search_key):
#     """Search for a key in a list of strings and return matching items.
# 
#     Parameters
#     ----------
#     str_list : list of str
#         The list of strings to search through.
#     search_key : str
#         The key to search for in the strings.
# 
#     Returns
#     -------
#     list
#         A list of strings from str_list that contain the search_key.
# 
#     Example
#     -------
#     >>> grep(['apple', 'banana', 'cherry'], 'a')
#     ['apple', 'banana']
#     >>> grep(['cat', 'dog', 'elephant'], 'e')
#     ['elephant']
#     """
#     """
#     Example:
#         str_list = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
#         search_key = 'orange'
#         print(grep(str_list, search_key))
#         # ([1, 5], ['orange', 'orange_juice'])
#     """
#     matched_keys = []
#     indi = []
#     for ii, string in enumerate(str_list):
#         m = re.search(search_key, string)
#         if m is not None:
#             matched_keys.append(string)
#             indi.append(ii)
#     return indi, matched_keys
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

from mngs.str._grep import *

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
