# src from here --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-07-11 06:06:21 (ywatanabe)"
# # /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_add_marginal_ax.py
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
# importlib.reload(mngs)
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
# from mpl_toolkits.axes_grid1 import make_axes_locatable
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
# # def add_marginal_axes(ax):
# #     divider = make_axes_locatable(ax)
# 
# #     ax_marg_x = divider.append_axes("top", size="20%", pad=0.1)
# #     ax_marg_x.set_box_aspect(0.2)
# 
# #     ax_marg_y = divider.append_axes("right", size="20%", pad=0.1)
# #     ax_marg_y.set_box_aspect(0.2 ** (-1))
# 
# #     return ax_marg_x, ax_marg_y
# 
# 
# def add_marginal_ax(ax, place, size=0.2, pad=0.1):
#     divider = make_axes_locatable(ax)
# 
#     size_perc_str = f"{size*100}%"
#     if place in ["left", "right"]:
#         size = 1.0 / size
# 
#     ax_marg = divider.append_axes(place, size=size_perc_str, pad=pad)
#     ax_marg.set_box_aspect(size)
# 
#     return ax_marg
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

from src.mngs.plt/ax/_add_marginal_ax.py import *

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
