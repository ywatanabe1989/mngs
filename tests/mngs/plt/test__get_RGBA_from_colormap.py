# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# 
# 
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib import cm
# 
# 
# class ColorGetter:
#     # https://stackoverflow.com/questions/26108436/how-can-i-get-the-matplotlib-rgb-color-given-the-colormap-name-boundrynorm-an
#     def __init__(self, cmap_name, start_val, stop_val):
#         self.cmap_name = cmap_name
#         self.cmap = plt.get_cmap(cmap_name)
#         self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
#         self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
# 
#     def get_rgb(self, val):
#         return self.scalarMap.to_rgba(val)
# 
# 
# def get_RGBA_from_colormap(val, cmap="Blues", cmap_start_val=0, cmap_stop_val=1):
#     ColGetter = ColorGetter(cmap, cmap_start_val, cmap_stop_val)
#     return ColGetter.get_rgb(val)

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

from src.mngs.plt/_get_RGBA_from_colormap.py import *

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
