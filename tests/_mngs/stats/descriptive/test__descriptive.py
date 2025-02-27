# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-16 03:59:50 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/stats/descriptive/_descriptive.py
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/stats/descriptive/_descriptive.py"
# 
# """
# Functionality:
#     - Computes descriptive statistics on PyTorch tensors
# Input:
#     - PyTorch tensor or numpy array
# Output:
#     - Descriptive statistics (mean, std, quantiles, etc.)
# Prerequisites:
#     - PyTorch, NumPy
# """
# 
# from typing import List, Optional, Tuple, Union
# 
# import numpy as np
# import torch
# 
# from ...decorators import torch_fn
# 
# 
# @torch_fn
# def describe(
#     x: torch.Tensor,
#     axis: int = -1,
#     dim: Optional[Union[int, Tuple[int, ...]]] = None,
#     keepdims: bool = False,
#     funcs: Union[List[str], str] = [
#         "nanmean",
#         "nanstd",
#         "nankurtosis",
#         "nanskewness",
#         "nanq25",
#         "nanq50",
#         "nanq75",
#         "nancount",
#     ],
#     device: Optional[torch.device] = None,
# ) -> Tuple[torch.Tensor, List[str]]:
#     """
#     Computes various descriptive statistics.
# 
#     Parameters
#     ----------
#     x : torch.Tensor
#         Input tensor
#     axis : int, default=-1
#         Deprecated. Use dim instead
#     dim : int or tuple of ints, optional
#         Dimension(s) along which to compute statistics
#     keepdims : bool, default=True
#         Whether to keep reduced dimensions
#     funcs : list of str or "all"
#         Statistical functions to compute
#     device : torch.device, optional
#         Device to use for computation
# 
#     Returns
#     -------
#     Tuple[torch.Tensor, List[str]]
#         Computed statistics and their names
#     """
#     dim = axis if dim is None else dim
#     dim = (dim,) if isinstance(dim, int) else tuple(dim)
# 
#     func_names = funcs
#     func_candidates = {
#         "mean": mean,
#         "std": std,
#         "kurtosis": kurtosis,
#         "skewness": skewness,
#         "q25": q25,
#         "q50": q50,
#         "q75": q75,
#         "nanmean": nanmean,
#         "nanstd": nanstd,
#         "nanvar": nanvar,
#         "nankurtosis": nankurtosis,
#         "nanskewness": nanskewness,
#         "nanq25": nanq25,
#         "nanq50": nanq50,
#         "nanq75": nanq75,
#         "nanmax": nanmax,
#         "nanmin": nanmin,
#         "nancount": nancount,
#         # "nanprod": nanprod,
#         # "nanargmin": nanargmin,
#         # "nanargmax": nanargmax,
#     }
# 
#     if funcs == "all":
#         _funcs = list(func_candidates.values())
#         func_names = list(func_candidates.keys())
#     else:
#         _funcs = [func_candidates[ff] for ff in func_names]
# 
#     calculated = [ff(x, dim=dim, keepdims=keepdims) for ff in _funcs]
#     return torch.stack(calculated, dim=-1), func_names
# 
# 
# @torch_fn
# def mean(x, axis=-1, dim=None, keepdims=False):
#     return x.mean(dim, keepdims=keepdims)
# 
# 
# @torch_fn
# def std(x, axis=-1, dim=None, keepdims=False):
#     return x.std(dim, keepdims=keepdims)
# 
# 
# @torch_fn
# def var(x, axis=-1, dim=None, keepdims=False):
#     return x.var(dim, keepdims=keepdims)
# 
# 
# @torch_fn
# def skewness(x, axis=-1, dim=None, keepdims=False):
#     zscores = zscore(x, axis=axis, keepdims=keepdims)
#     return torch.mean(torch.pow(zscores, 3.0), dim=dim, keepdims=keepdims)
# 
# 
# @torch_fn
# def kurtosis(x, axis=-1, dim=None, keepdims=False):
#     zscores = zscore(x, axis=axis, keepdims=keepdims)
#     return (
#         torch.mean(torch.pow(zscores, 4.0), dim=dim, keepdims=keepdims) - 3.0
#     )
# 
# 
# @torch_fn
# def zscore(x, axis=-1, dim=None, keepdims=False):
#     _mean = mean(x, dim=dim, keepdims=keepdims)
#     _std = std(x, dim=dim, keepdims=keepdims)
#     return (x - _mean) / _std
# 
# 
# @torch_fn
# def quantile(x, q, axis=-1, dim=None, keepdims=False):
#     return torch.quantile(x, q / 100, dim=dim, keepdims=keepdims)
# 
# 
# @torch_fn
# def quantile(x, q, axis=-1, dim=None, keepdims=False):
#     dim = axis if dim is None else dim
#     if isinstance(dim, (tuple, list)):
#         for d in sorted(dim, reverse=True):
#             x = torch.quantile(x, q / 100, dim=d, keepdims=keepdims)
#     else:
#         x = torch.quantile(x, q / 100, dim=dim, keepdims=keepdims)
#     return x
# 
# 
# @torch_fn
# def q25(x, axis=-1, dim=None, keepdims=False):
#     return quantile(x, 25, axis=axis, dim=dim, keepdims=keepdims)
# 
# 
# @torch_fn
# def q50(x, axis=-1, dim=None, keepdims=False):
#     return quantile(x, 50, axis=axis, dim=dim, keepdims=keepdims)
# 
# 
# @torch_fn
# def q75(x, axis=-1, dim=None, keepdims=False):
#     return quantile(x, 75, axis=axis, dim=dim, keepdims=keepdims)
# 
# 
# @torch_fn
# def nanmax(x, axis=-1, dim=None, keepdims=False):
#     min_value = torch.finfo(x.dtype).min
#     dim = axis if dim is None else dim
#     if isinstance(dim, (tuple, list)):
#         for d in sorted(dim, reverse=True):
#             x = x.nan_to_num(min_value).max(dim=d, keepdims=keepdims)[0]
#     else:
#         x = x.nan_to_num(min_value).max(dim=dim, keepdims=keepdims)[0]
#     return x
# 
# 
# @torch_fn
# def nanmin(x, axis=-1, dim=None, keepdims=False):
#     max_value = torch.finfo(x.dtype).max
#     dim = axis if dim is None else dim
#     if isinstance(dim, (tuple, list)):
#         for d in sorted(dim, reverse=True):
#             x = x.nan_to_num(max_value).min(dim=d, keepdims=keepdims)[0]
#     else:
#         x = x.nan_to_num(max_value).min(dim=dim, keepdims=keepdims)[0]
#     return x
# 
# 
# @torch_fn
# def nanmean(x, axis=-1, dim=None, keepdims=False):
#     return torch.nanmean(x, dim=dim, keepdims=keepdims)
# 
# 
# # @torch_fn
# # def nanvar(x, axis=-1, dim=None, keepdims=False):
# #     tensor_mean = x.nanmean(dim=dim, keepdims=True)
# #     return (x - tensor_mean).square().nanmean(dim=dim, keepdims=keepdims)
# 
# 
# @torch_fn
# def nanvar(x, axis=-1, dim=None, keepdims=False):
#     tensor_mean = nanmean(x, dim=dim, keepdims=True)
#     return (x - tensor_mean).square().nanmean(dim=dim, keepdims=keepdims)
# 
# 
# @torch_fn
# def nanstd(x, axis=-1, dim=None, keepdims=False):
#     return torch.sqrt(nanvar(x, dim=dim, keepdims=keepdims))
# 
# 
# @torch_fn
# def nanzscore(x, axis=-1, dim=None, keepdims=False):
#     _mean = nanmean(x, dim=dim, keepdims=keepdims)
#     _std = nanstd(x, dim=dim, keepdims=keepdims)
#     return (x - _mean) / _std
# 
# 
# @torch_fn
# def nankurtosis(x, axis=-1, dim=None, keepdims=False):
#     zscores = nanzscore(x, axis=axis, keepdims=keepdims)
#     return (
#         torch.nanmean(torch.pow(zscores, 4.0), dim=dim, keepdims=keepdims) - 3.0
#     )
# 
# 
# @torch_fn
# def nanskewness(x, axis=-1, dim=None, keepdims=False):
#     zscores = nanzscore(x, axis=axis, keepdims=keepdims)
#     return torch.nanmean(torch.pow(zscores, 3.0), dim=dim, keepdims=keepdims)
# 
# 
# @torch_fn
# def nanprod(x, axis=-1, dim=None, keepdims=False):
#     dim = axis if dim is None else dim
#     if isinstance(dim, (tuple, list)):
#         for d in sorted(dim, reverse=True):
#             x = x.nan_to_num(1).prod(dim=d, keepdims=keepdims)
#     else:
#         x = x.nan_to_num(1).prod(dim=dim, keepdims=keepdims)
#     return x
# 
# 
# @torch_fn
# def nancumprod(x, axis=-1, dim=None, keepdims=False):
#     dim = axis if dim is None else dim
#     if isinstance(dim, (tuple, list)):
#         raise ValueError("cumprod does not support multiple dimensions")
#     return x.nan_to_num(1).cumprod(dim=dim)
# 
# 
# @torch_fn
# def nancumsum(x, axis=-1, dim=None, keepdims=False):
#     dim = axis if dim is None else dim
#     if isinstance(dim, (tuple, list)):
#         raise ValueError("cumsum does not support multiple dimensions")
#     return x.nan_to_num(0).cumsum(dim=dim)
# 
# 
# @torch_fn
# def nanargmin(x, axis=-1, dim=None, keepdims=False):
#     max_value = torch.finfo(x.dtype).max
#     dim = axis if dim is None else dim
#     if isinstance(dim, (tuple, list)):
#         for d in sorted(dim, reverse=True):
#             x = x.nan_to_num(max_value).argmin(dim=d, keepdims=keepdims)
#     else:
#         x = x.nan_to_num(max_value).argmin(dim=dim, keepdims=keepdims)
#     return x
# 
# 
# @torch_fn
# def nanargmax(x, axis=-1, dim=None, keepdims=False):
#     min_value = torch.finfo(x.dtype).min
#     dim = axis if dim is None else dim
#     if isinstance(dim, (tuple, list)):
#         for d in sorted(dim, reverse=True):
#             x = x.nan_to_num(min_value).argmax(dim=d, keepdims=keepdims)
#     else:
#         x = x.nan_to_num(min_value).argmax(dim=dim, keepdims=keepdims)
#     return x
# 
# 
# @torch_fn
# def nanquantile(x, q, axis=-1, dim=None, keepdims=False):
#     dim = axis if dim is None else dim
#     if isinstance(dim, (tuple, list)):
#         for d in sorted(dim, reverse=True):
#             mask = ~torch.isnan(x)
#             x_filtered = torch.where(mask, x, torch.tensor(float("inf")))
#             x = torch.quantile(x_filtered, q / 100, dim=d, keepdims=keepdims)
#     else:
#         mask = ~torch.isnan(x)
#         x_filtered = torch.where(mask, x, torch.tensor(float("inf")))
#         x = torch.quantile(x_filtered, q / 100, dim=dim, keepdims=keepdims)
#     return x
# 
# 
# @torch_fn
# def nanq25(x, axis=-1, dim=None, keepdims=False):
#     return nanquantile(x, 25, axis=axis, dim=dim, keepdims=keepdims)
# 
# 
# @torch_fn
# def nanq50(x, axis=-1, dim=None, keepdims=False):
#     return nanquantile(x, 50, axis=axis, dim=dim, keepdims=keepdims)
# 
# 
# @torch_fn
# def nanq75(x, axis=-1, dim=None, keepdims=False):
#     return nanquantile(x, 75, axis=axis, dim=dim, keepdims=keepdims)
# 
# 
# @torch_fn
# def nancount(x, axis=-1, dim=None, keepdims=False):
#     """Count number of non-NaN values along specified dimensions.
# 
#     Parameters
#     ----------
#     x : torch.Tensor
#         Input tensor
#     axis : int, default=-1
#         Deprecated. Use dim instead
#     dim : int or tuple of ints, optional
#         Dimension(s) along which to count
#     keepdims : bool, default=True
#         Whether to keep reduced dimensions
# 
#     Returns
#     -------
#     torch.Tensor
#         Count of non-NaN values
#     """
#     dim = axis if dim is None else dim
#     mask = ~torch.isnan(x)
# 
#     if isinstance(dim, (tuple, list)):
#         for d in sorted(dim, reverse=True):
#             mask = mask.sum(dim=d, keepdims=keepdims)
#     else:
#         mask = mask.sum(dim=dim, keepdims=keepdims)
# 
#     return mask
# 
# 
# if __name__ == "__main__":
#     # from mngs.stats.descriptive import *
# 
#     x = np.random.rand(4, 3, 2)
#     print(describe(x, dim=(1, 2), keepdims=False)[0].shape)
#     print(describe(x, funcs="all", dim=(1, 2), keepdims=False)[0].shape)
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

from mngs..stats.descriptive._descriptive import *

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
