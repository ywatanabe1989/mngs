# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 21:25:50 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_mv_to_tmp.py
# 
# from shutil import move
# 
# def _mv_to_tmp(fpath, L=2):
#     try:
#         tgt_fname = "-".join(fpath.split("/")[-L:])
#         tgt_fpath = "/tmp/{}".format(tgt_fname)
#         move(fpath, tgt_fpath)
#         print("Moved to: {}".format(tgt_fpath))
#     except:
#         pass
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

from ...src.mngs..io._mv_to_tmp import *

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
