# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# 
# import numpy as np
# 
# 
# def calc_partial_corr(x, y, z):
#     """remove the influence of the variable z from the correlation between x and y."""
# 
#     x = np.array(x).astype(np.float128)
#     y = np.array(y).astype(np.float128)
#     z = np.array(z).astype(np.float128)
# 
#     r_xy = np.corrcoef(x, y)[0, 1]
#     r_xz = np.corrcoef(x, z)[0, 1]
#     r_yz = np.corrcoef(y, z)[0, 1]
#     r_xy_z = (r_xy - r_xz * r_yz) / (np.sqrt(1 - r_xz ** 2) * np.sqrt(1 - r_yz ** 2))
#     return r_xy_z

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

from src.mngs.stats._calc_partial_corr import *

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
