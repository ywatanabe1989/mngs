# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# 
# import numpy as np
# from scipy import stats
# 
# 
# def calc_partial_corrcoef(x, y, z):
#     """remove the influence of the variable z from the correlation between x and y."""
#     r_xy = np.corrcoef(x, y)
#     r_xz = np.corrcoef(x, z)
#     r_yz = np.corrcoef(y, z)
#     r_xy_z = (r_xy - r_xz * r_yz) / (1 - r_xz ** 2) * (1 - r_yz ** 2)
#     return r_xy_z
# 
# 
# def nocorrelation_test(x, y, z=None, alpha=0.05):
#     if z is None:
#         r = np.corrcoef(x, y)[1, 0]
#     if z is not None:
#         r = calc_partial_corrcoef(x, y, z)[1, 0]
# 
#     n = len(x)
#     df = n - 2
#     # t = np.abs(np.array(r)) * np.sqrt((df) / (1 - np.array(r)**2))
#     t = np.abs(r) * np.sqrt((df) / (1 - r ** 2))
#     # t_alpha = scipy.stats.t.ppf(1 - alpha / 2, df)
#     p_value = 2 * (1 - stats.t.cdf(t, df))
#     return r, t, p_value

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

from mngs..stats.tests._nocorrelation_test import *

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
