# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:43:47 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_preserve_docstring.py
# 
# from functools import wraps
# def preserve_doc(loader_func):
#     """Wrap the loader functions to preserve their docstrings"""
#     @wraps(loader_func)
#     def wrapper(*args, **kwargs):
#         return loader_func(*args, **kwargs)
# 
#     return wrapper
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

from mngs..decorators._preserve_docstring import *

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
