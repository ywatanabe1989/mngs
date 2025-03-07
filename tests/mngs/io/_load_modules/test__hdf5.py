# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:55:37 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_load_modules/_hdf5.py
# 
# from typing import Any
# 
# import h5py
# 
# 
# def _load_hdf5(lpath: str, **kwargs) -> Any:
#     """Load HDF5 file."""
#     if not lpath.endswith(".hdf5"):
#         raise ValueError("File must have .hdf5 extension")
#     obj = {}
#     with h5py.File(lpath, "r") as hf:
#         for name in hf:
#             obj[name] = hf[name][:]
#     return obj
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

from mngs..io._load_modules._hdf5 import *

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
