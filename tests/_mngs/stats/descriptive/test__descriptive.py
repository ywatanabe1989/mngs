# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-03 03:25:43 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/stats/descriptive/_descriptive.py
# 
# import mngs
# import numpy as np
# import torch
# from ...decorators import torch_fn
# 
# @torch_fn
# def mean(x, axis=-1, dim=None, keepdims=True):
#     return x.mean(axis, keepdims=keepdims)
# 
# @torch_fn
# def std(x, axis=-1, dim=None, keepdims=True):
#     return x.std(axis, keepdims=keepdims)
# 
# 
# @torch_fn
# def zscore(x, axis=-1, dim=None, keepdims=True):
#     _mean = mean(x, axis=axis)
#     diffs = x - _mean
#     var = torch.mean(torch.pow(diffs, 2.0), dim=axis, keepdims=True)
#     std = torch.pow(var, 0.5)
#     return diffs / std
# 
# 
# @torch_fn
# def kurtosis(x, axis=-1, dim=None, keepdims=True):
#     zscores = zscore(x, axis=axis)
#     return torch.mean(torch.pow(zscores, 4.0), dim=axis, keepdims=True) - 3.0
# 
# 
# @torch_fn
# def skewness(x, axis=-1, dim=None, keepdims=True):
#     zscores = zscore(x, axis=axis)
#     return torch.mean(torch.pow(zscores, 3.0), dim=axis, keepdims=True)
# 
# 
# @torch_fn
# def median(x, axis=-1, dim=None, keepdims=True):
#     return torch.median(x, dim=axis, keepdims=True)[0]
# 
# 
# @torch_fn
# def q(x, q, axis=-1, dim=None, keepdims=True):
#     return torch.quantile(x, q / 100, dim=axis, keepdims=True)
# 
# 
# if __name__ == "__main__":
#     from mngs.stats.descriptive import *
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

from mngs.stats.descriptive._descriptive import *

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
