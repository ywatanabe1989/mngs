# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-28 19:38:19 (ywatanabe)"
# # /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/torch/_nan_funcs.py
# 
# import torch as _torch
# 
# # https://github.com/pytorch/pytorch/issues/61474
# def nanmax(tensor, dim=None, keepdim=False):
#     min_value = _torch.finfo(tensor.dtype).min
#     output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
#     return output
# 
# 
# def nanmin(tensor, dim=None, keepdim=False):
#     max_value = _torch.finfo(tensor.dtype).max
#     output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)
#     return output
# 
# 
# def nanvar(tensor, dim=None, keepdim=False):
#     tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
#     output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
#     return output
# 
# 
# def nanstd(tensor, dim=None, keepdim=False):
#     output = nanvar(tensor, dim=dim, keepdim=keepdim)
#     output = output.sqrt()
#     return output
# 
# 
# def nanprod(tensor, dim=None, keepdim=False):
#     output = tensor.nan_to_num(1).prod(dim=dim, keepdim=keepdim)
#     return output
# 
# 
# def nancumprod(tensor, dim=None, keepdim=False):
#     output = tensor.nan_to_num(1).cumprod(dim=dim, keepdim=keepdim)
#     return output
# 
# 
# def nancumsum(tensor, dim=None, keepdim=False):
#     output = tensor.nan_to_num(0).cumsum(dim=dim, keepdim=keepdim)
#     return output
# 
# 
# def nanargmin(tensor, dim=None, keepdim=False):
#     max_value = _torch.finfo(tensor.dtype).max
#     output = tensor.nan_to_num(max_value).argmin(dim=dim, keepdim=keepdim)
#     return output
# 
# 
# def nanargmax(tensor, dim=None, keepdim=False):
#     min_value = _torch.finfo(tensor.dtype).min
#     output = tensor.nan_to_num(min_value).argmax(dim=dim, keepdim=keepdim)
#     return output

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

from ...src.mngs..torch._nan_funcs import *

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
