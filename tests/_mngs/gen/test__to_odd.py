# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 04:02:32 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/gen/_to_odd.py
# def to_odd(n):
#     """Convert a number to the nearest odd number less than or equal to itself.
# 
#     Parameters
#     ----------
#     n : int or float
#         The input number to be converted.
# 
#     Returns
#     -------
#     int
#         The nearest odd number less than or equal to the input.
# 
#     Example
#     -------
#     >>> to_odd(6)
#     5
#     >>> to_odd(7)
#     7
#     >>> to_odd(5.8)
#     5
#     """
#     return int(n) - ((int(n) + 1) % 2)
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

from mngs.gen._to_odd import *

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
