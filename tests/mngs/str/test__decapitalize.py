# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 11:25:00 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/str/_decapitalize.py
# 
# """
# Functionality:
#     - Converts first character of string to lowercase
# Input:
#     - String to be processed
# Output:
#     - Modified string with lowercase first character
# Prerequisites:
#     - None
# """
# 
# 
# def decapitalize(input_string: str) -> str:
#     """Converts first character of string to lowercase.
# 
#     Example
#     -------
#     >>> decapitalize("Hello")
#     'hello'
#     >>> decapitalize("WORLD")
#     'wORLD'
#     >>> decapitalize("")
#     ''
# 
#     Parameters
#     ----------
#     input_string : str
#         String to be processed
# 
#     Returns
#     -------
#     str
#         Modified string with first character in lowercase
# 
#     Raises
#     ------
#     TypeError
#         If input is not a string
#     """
#     try:
#         if not isinstance(input_string, str):
#             raise TypeError("Input must be a string")
# 
#         if not input_string:
#             return input_string
# 
#         return input_string[0].lower() + input_string[1:]
# 
#     except Exception as error:
#         raise ValueError(f"String processing failed: {str(error)}")
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

from ...src.mngs..str._decapitalize import *

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
