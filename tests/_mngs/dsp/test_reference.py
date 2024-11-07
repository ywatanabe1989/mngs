# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "ywatanabe (2024-11-02 22:48:44)"
# # File: ./mngs_repo/src/mngs/dsp/reference.py
# 
# import torch as _torch
# from ..decorators import torch_fn as _torch_fn
# 
# 
# @_torch_fn
# def common_average(x, dim=-2):
#     re_referenced = (x - x.mean(dim=dim, keepdims=True)) / x.std(
#         dim=dim, keepdims=True
#     )
#     assert x.shape == re_referenced.shape
#     return re_referenced
# 
# @_torch_fn
# def random(x, dim=-2):
#     idx_all = [slice(None)] * x.ndim
#     idx_rand_dim = _torch.randperm(x.shape[dim])
#     idx_all[dim] = idx_rand_dim
#     y = x[idx_all]
#     re_referenced = x - y
#     assert x.shape == re_referenced.shape
#     return re_referenced
# 
# @_torch_fn
# def take_reference(x, tgt_indi, dim=-2):
#     idx_all = [slice(None)] * x.ndim
#     idx_all[dim] = tgt_indi
#     re_referenced = x - x[tgt_indi]
#     assert x.shape == re_referenced.shape
#     return re_referenced
# 
# if __name__ == "__main__":
#     import mngs
# 
#     x, f, t = mngs.dsp.demo_sig()
#     y = common_average(x)
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
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.dsp.reference import *

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
