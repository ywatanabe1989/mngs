# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:55:43 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_load_modules/_matlab.py
# 
# from typing import Any
# 
# from pymatreader import read_mat
# 
# 
# def _load_matlab(lpath: str, **kwargs) -> Any:
#     """Load MATLAB file."""
#     if not lpath.endswith(".mat"):
#         raise ValueError("File must have .mat extension")
#     return read_mat(lpath, **kwargs)
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

from mngs..io._load_modules._matlab import *

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
