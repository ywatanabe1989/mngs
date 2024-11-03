# src from here --------------------------------------------------------------------------------
# import mngs
# import numpy as np
# import matplotlib
# # matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# 
# 
# def mk_colorbar(start="white", end="blue"):
#     xx = np.linspace(0, 1, 256)
# 
#     start = np.array(mngs.plt.colors.RGB_d[start])
#     end = np.array(mngs.plt.colors.RGB_d[end])
#     colors = (end-start)[:, np.newaxis]*xx
# 
#     colors -= colors.min()
#     colors /= colors.max()
# 
#     fig, ax = plt.subplots()
#     [ax.axvline(_xx, color=colors[:,i_xx]) for i_xx, _xx in enumerate(xx)]
#     ax.xaxis.set_ticks_position("none")
#     ax.yaxis.set_ticks_position("none")
#     ax.set_aspect(0.2)
#     return fig
# 
# 

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

from src.mngs.plt._mk_colorbar import *

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
