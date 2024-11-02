# src from here --------------------------------------------------------------------------------
# import os
# from contextlib import contextmanager, redirect_stderr, redirect_stdout
# 
# @contextmanager
# def suppress_output(suppress=True):
#     """
#     A context manager that suppresses stdout and stderr.
# 
#     Example:
#         with suppress_output():
#             print("This will not be printed to the console.")
#     """
#     if suppress:
#         # Open a file descriptor that points to os.devnull (a black hole for data)
#         with open(os.devnull, "w") as fnull:
#             # Temporarily redirect stdout and stderr to the file descriptor fnull
#             with redirect_stdout(fnull), redirect_stderr(fnull):
#                 # Yield control back to the context block
#                 yield
#     else:
#         # If suppress is False, just yield without redirecting output
#         yield
# 
# quiet = suppress_output

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

from src.mngs.gen/_suppress_output.py import *

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
