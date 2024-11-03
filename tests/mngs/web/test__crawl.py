# src from here --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-07-29 22:30:22 (ywatanabe)"
# # /home/ywatanabe/proj/mngs_repo/src/mngs/web/_crawl.py
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
# import os
# import re
# import sys
# 
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# import importlib
# 
# import mngs
# 
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from icecream import ic
# from natsort import natsorted
# from glob import glob
# from pprint import pprint
# import warnings
# import logging
# from tqdm import tqdm
# import xarray as xr
# 
# # sys.path = ["."] + sys.path
# # from scripts import utils, load
# 
# """
# Warnings
# """
# # warnings.simplefilter("ignore", UserWarning)
# 
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
# def main():
#     pass
# 
# 
# if __name__ == "__main__":
#     # # Argument Parser
#     # import argparse
#     # parser = argparse.ArgumentParser(description='')
#     # parser.add_argument('--var', '-v', type=int, default=1, help='')
#     # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
#     # args = parser.parse_args()
# 
#     # Main
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

from src.mngs.web._crawl import *

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
