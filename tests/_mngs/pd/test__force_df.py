# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-09-29 05:55:52 (ywatanabe)"
# # /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/pd/_force_df.py
# 
# import numpy as np
# import pandas as pd
# import mngs
# 
# # def force_df(permutable_dict, filler=""):
# def force_df(permutable_dict, filler=np.nan):
# 
#     if mngs.gen.is_listed_X(permutable_dict, pd.Series):
#         permutable_dict = [sr.to_dict() for sr in permutable_dict]
#     ## Deep copy
#     permutable_dict = permutable_dict.copy()
# 
#     ## Get the lengths
#     max_len = 0
#     for k, v in permutable_dict.items():
#         # Check if v is an iterable (but not string) or treat as single length otherwise
#         if isinstance(v, (str, int, float)) or not hasattr(v, "__len__"):
#             length = 1
#         else:
#             length = len(v)
#         max_len = max(max_len, length)
# 
#     ## Replace with appropriately filled list
#     for k, v in permutable_dict.items():
#         if isinstance(v, (str, int, float)) or not hasattr(v, "__len__"):
#             permutable_dict[k] = [v] + [filler] * (max_len - 1)
#         else:
#             permutable_dict[k] = list(v) + [filler] * (max_len - len(v))
# 
#     ## Puts them into a DataFrame
#     out_df = pd.DataFrame(permutable_dict)
# 
#     return out_df

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

from mngs.pd._force_df import *

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