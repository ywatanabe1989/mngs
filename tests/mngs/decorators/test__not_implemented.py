# src from here --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-07 22:16:25 (ywatanabe)"
# # /home/ywatanabe/proj/mngs/src/mngs/gen/_not_implemented.py
# 
# import warnings
# 
# 
# def not_implemented(func):
#     """
#     Decorator to mark methods as not implemented, issue a warning, and prevent their execution.
# 
#     Arguments:
#         func (callable): The function or method to decorate.
# 
#     Returns:
#         callable: A wrapper function that issues a warning and raises NotImplementedError when called.
#     """
# 
#     def wrapper(*args, **kwargs):
#         # Issue a warning before raising the error
#         warnings.warn(
#             f"Attempt to use unimplemented method: '{func.__name__}'. This method is not yet available.",
#             category=FutureWarning,
#             stacklevel=2,
#         )
#         # # Raise the NotImplementedError
#         # raise NotImplementedError(f"The method '{func.__name__}' is not implemented yet.")
# 
#     return wrapper

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

from ...src.mngs..decorators._not_implemented import *

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
