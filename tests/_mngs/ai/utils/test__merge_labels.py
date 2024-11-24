# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# 
# import mngs
# import numpy as np
# 
# # y1, y2 = T_tra, M_tra
# # def merge_labels(y1, y2):
# #     y = [str(z1) + "-" + str(z2) for z1, z2 in zip(y1, y2)]
# #     conv_d = {z: i for i, z in enumerate(np.unique(y))}
# #     y = [conv_d[z] for z in y]
# #     return y
# 
# 
# def merge_labels(*ys, to_int=False):
#     if not len(ys) > 1:  # Check if more than two arguments are passed
#         return ys[0]
#     else:
#         y = [mngs.gen.connect_nums(zs) for zs in zip(*ys)]
#         if to_int:
#             conv_d = {z: i for i, z in enumerate(np.unique(y))}
#             y = [conv_d[z] for z in y]
#         return np.array(y)

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

from mngs..ai.utils._merge_labels import *

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
