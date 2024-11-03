# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-03-30 18:17:46 (ywatanabe)"
# 
# import mngs
# import numpy as np
# import torch
# 
# 
# def mean(x, axis=-1, dim=None):
#     if dim is not None:
#         axis = dim
#     return x.mean(axis, keepdims=True)
# 
# 
# def std(x, axis=-1, dim=None):
#     if dim is not None:
#         axis = dim
#     return x.std(axis, keepdims=True)
# 
# 
# def zscore(x, axis=-1, dim=None):
#     if dim is not None:
#         axis = dim
# 
#     x, x_type = mngs.gen.my2tensor(x)
# 
#     _mean = mean(x, axis=axis)
#     diffs = x - _mean
#     var = torch.mean(torch.pow(diffs, 2.0), dim=axis, keepdims=True)
#     std = torch.pow(var, 0.5)
#     out = diffs / std
# 
#     if x_type == "numpy":
#         return mngs.gen.my2array(out)[0]
#     else:
#         return out
# 
# 
# def kurtosis(x, axis=-1, dim=None):
#     if dim is not None:
#         axis = dim
# 
#     x, x_type = mngs.gen.my2tensor(x)
# 
#     zscores = zscore(x, axis=axis)
#     out = torch.mean(torch.pow(zscores, 4.0), dim=axis, keepdims=True) - 3.0
# 
#     if x_type == "numpy":
#         return mngs.gen.my2array(out)[0]
#     else:
#         return out
# 
# 
# def skewness(x, axis=-1, dim=None):
#     if dim is not None:
#         axis = dim
# 
#     x, x_type = mngs.gen.my2tensor(x)
# 
#     zscores = zscore(x, dim=axis)
#     out = torch.mean(torch.pow(zscores, 3.0), dim=axis, keepdims=True)
# 
#     if x_type == "numpy":
#         return mngs.gen.my2array(out)[0]
#     else:
#         return out
# 
# 
# def median(x, axis=-1, dim=None):
#     if dim is not None:
#         axis = dim
# 
#     x, x_type = mngs.gen.my2tensor(x)
# 
#     out = torch.median(x, dim=axis, keepdims=True)[0]
# 
#     if x_type == "numpy":
#         return mngs.gen.my2array(out)[0]
#     else:
#         return out
# 
# 
# def q(x, q, axis=-1, dim=None):
#     if dim is not None:
#         axis = dim
# 
#     x, x_type = mngs.gen.my2tensor(x)
# 
#     out = torch.quantile(x, q / 100, dim=axis, keepdims=True)
# 
#     if x_type == "numpy":
#         return mngs.gen.my2array(out)[0]
#     else:
#         return out
# 
# 
# if __name__ == "__main__":
#     x = np.random.rand(4, 3, 2)
#     mean(x)
#     std(x)
#     zscore(x)
#     kurtosis(x)
#     skewness(x)
#     median(x)
#     q(x, 25)
# 
#     x = torch.tensor(x)
#     mean(x)
#     std(x)
#     zscore(x)
#     kurtosis(x)
#     skewness(x)
#     median(x)
#     q(x, 25)

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

from src.mngs.stats/_general.py import *

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
