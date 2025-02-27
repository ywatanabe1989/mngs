# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:51:45 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_load_modules/_con.py
# 
# from typing import Any
# 
# import mne
# 
# 
# def _load_con(lpath: str, **kwargs) -> Any:
#     if not lpath.endswith(".con"):
#         raise ValueError("File must have .con extension")
#     obj = mne.io.read_raw_fif(lpath, preload=True, **kwargs)
#     obj = obj.to_data_frame()
#     obj["samp_rate"] = obj.info["sfreq"]
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

from ...src.mngs..io._load_modules._con import *

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
