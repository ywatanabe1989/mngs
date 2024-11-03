# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-05 12:14:08 (ywatanabe)"
# 
# from ..decorators import torch_fn
# 
# 
# @torch_fn
# def ensure_3d(x):
#     if x.ndim == 1:  # assumes (seq_len,)
#         x = x.unsqueeze(0).unsqueeze(0)
#     elif x.ndim == 2:  # assumes (batch_siize, seq_len)
#         x = x.unsqueeze(1)
#     return x
# 
# 
# # @torch_fn
# # def unbias(x, dim=-1, fn="mean"):
# #     if fn == "mean":
# #         return x - x.mean(dim=dim, keepdims=True)
# #     if fn == "min":
# #         return x - x.min(dim=dim, keepdims=True)[0]

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

from src.mngs.dsp._misc import *

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
