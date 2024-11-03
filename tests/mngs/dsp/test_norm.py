# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-05 12:15:42 (ywatanabe)"
# 
# import torch as _torch
# from ..decorators import torch_fn as _torch_fn
# 
# 
# @_torch_fn
# def z(x, dim=-1):
#     return (x - x.mean(dim=dim, keepdim=True)) / x.std(dim=dim, keepdim=True)
# 
# 
# @_torch_fn
# def minmax(x, amp=1.0, dim=-1, fn="mean"):
#     MM = x.max(dim=dim, keepdims=True)[0].abs()
#     mm = x.min(dim=dim, keepdims=True)[0].abs()
#     return amp * x / _torch.maximum(MM, mm)

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

from src.mngs.dsp.norm import *

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
