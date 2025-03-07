# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-24 17:17:05 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/str/_print_debug.py
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/str/_print_debug.py"
# 
# from ._printc import printc
# 
# 
# def print_debug():
#     printc(
#         (
#             f"{'!'*60}\n"
#             f"{'!'*60}\n"
#             f"{'!'*60}\n"
#             f"{'!'*60}\n"
#             f"{'!'*60}\n"
#             f"{'!'*60}\n"
#             f"{'!'*24} DEBUG MODE {'!'*24}\n"
#             f"{'!'*60}\n"
#             f"{'!'*60}\n"
#             f"{'!'*60}\n"
#             f"{'!'*60}\n"
#             f"{'!'*60}\n"
#             f"{'!'*60}"
#         ),
#         c="yellow",
#         char="!",
#         n=60,
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

from mngs..str._print_debug import *

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
