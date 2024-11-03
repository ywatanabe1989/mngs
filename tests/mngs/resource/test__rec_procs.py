# src from here --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-05-16 13:19:50 (ywatanabe)"
# # Author: Yusuke Watanabe (ywata1989@gmail.com)
# 
# 
# """
# This script does XYZ.
# """
# 
# 
# """
# Imports
# """
# import math
# import os
# import sys
# import time
# 
# import matplotlib.pyplot as plt
# import mngs
# import pandas as pd
# 
# """
# Config
# """
# # CONFIG = mngs.gen.load_configs()
# 
# 
# """
# Functions & Classes
# """
# 
# 
# def main(
#     path="/tmp/mngs/processer_usages.csv",
#     limit_min=3,
#     interval_s=1,
#     reset=True,
#     verbose=True,
# ):
#     # Parameters
#     limit_s = limit_min * 60
#     n_max = math.ceil(limit_s // interval_s)
# 
#     if reset and os.path.exists(path):
#         mngs.sh(f"rm {path}")
#         mngs.io.save(pd.DataFrame(), path, verbose=False)
#         print(f"\n{path} was cleared.")
# 
#     for _ in range(n_max):
#         add(path, verbose=verbose)
#         time.sleep(interval_s)
# 
# 
# def add(path, verbose=True):
#     try:
#         past = mngs.io.load(path)
#     except Exception as e:
#         print(e)
#         past = pd.DataFrame()
# 
#     now = mngs.res.get_proc_usages()
# 
#     combined = pd.concat([past, now]).round(3)
# 
#     mngs.io.save(combined, path, verbose=verbose)
# 
#     if verbose:
#         print(f"\n{combined}")
# 
# 
# rec_procs = main
# 
# if __name__ == "__main__":
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False
#     )
#     main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)
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
    sys.path.insert(0, project_root)

from src.mngs.resource._rec_procs import *

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
