#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-06 17:36:02 (ywatanabe)"
# File: ./mngs_repo/src/mngs/stats/descriptive/_descriptive.py

"""
Functionality:
    - Computes descriptive statistics on PyTorch tensors
Input:
    - PyTorch tensor or numpy array
Output:
    - Descriptive statistics (mean, std, quantiles, etc.)
Prerequisites:
    - PyTorch, NumPy
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from ...decorators import torch_fn

@torch_fn
def describe(
    x: torch.Tensor,
    axis: int = -1,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = True,
    funcs: Union[List[str], str] = [
        "mean",
        "std",
        "kurtosis",
        "skewness",
        "q25",
        "q50",
        "q75",
    ],
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Computes various descriptive statistics.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    axis : int, default=-1
        Deprecated. Use dim instead
    dim : int or tuple of ints, optional
        Dimension(s) along which to compute statistics
    keepdims : bool, default=True
        Whether to keep reduced dimensions
    funcs : list of str or "all"
        Statistical functions to compute
    device : torch.device, optional
        Device to use for computation

    Returns
    -------
    Tuple[torch.Tensor, List[str]]
        Computed statistics and their names
    """
    dim = axis if dim is None else dim
    dim = (dim,) if isinstance(dim, int) else tuple(dim)

    func_names = funcs
    func_candidates = {
        "mean": mean,
        "std": std,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "nanmean": nanmean,
        "nanstd": nanstd,
        "nanvar": nanvar,
        "nankurtosis": nankurtosis,
        "nanskewness": nanskewness,
        "nanq25": nanq25,
        "nanq50": nanq50,
        "nanq75": nanq75,
        "nanmax": nanmax,
        "nanmin": nanmin,
        "nanprod": nanprod,
        "nanargmin": nanargmin,
        "nanargmax": nanargmax,
    }

    if funcs == "all":
        _funcs = list(func_candidates.values())
        func_names = list(func_candidates.keys())
    else:
        _funcs = [func_candidates[ff] for ff in func_names]

    calculated = [ff(x, dim=dim, keepdims=keepdims) for ff in _funcs]
    return torch.stack(calculated, dim=-1), func_names


@torch_fn
def mean(x, axis=-1, dim=None, keepdims=True):
    return x.mean(dim, keepdims=keepdims)


@torch_fn
def std(x, axis=-1, dim=None, keepdims=True):
    return x.std(dim, keepdims=keepdims)


@torch_fn
def var(x, axis=-1, dim=None, keepdims=True):
    return x.var(dim, keepdims=keepdims)


@torch_fn
def skewness(x, axis=-1, dim=None, keepdims=True):
    zscores = zscore(x, axis=axis, keepdims=keepdims)
    return torch.mean(torch.pow(zscores, 3.0), dim=dim, keepdims=keepdims)


@torch_fn
def kurtosis(x, axis=-1, dim=None, keepdims=True):
    zscores = zscore(x, axis=axis, keepdims=keepdims)
    return (
        torch.mean(torch.pow(zscores, 4.0), dim=dim, keepdims=keepdims) - 3.0
    )


@torch_fn
def zscore(x, axis=-1, dim=None, keepdims=True):
    _mean = mean(x, dim=dim, keepdims=keepdims)
    _std = std(x, dim=dim, keepdims=keepdims)
    return (x - _mean) / _std


@torch_fn
def median(x, axis=-1, dim=None, keepdims=True):
    return torch.median(x, dim=dim, keepdims=keepdims)[0]


@torch_fn
def quantile(x, q, axis=-1, dim=None, keepdims=True):
    return torch.quantile(x, q / 100, dim=dim, keepdims=keepdims)


@torch_fn
def quantile(x, q, axis=-1, dim=None, keepdims=True):
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        for d in sorted(dim, reverse=True):
            x = torch.quantile(x, q / 100, dim=d, keepdim=keepdims)
    else:
        x = torch.quantile(x, q / 100, dim=dim, keepdim=keepdims)
    return x


@torch_fn
def q25(x, axis=-1, dim=None, keepdims=True):
    return quantile(x, 25, axis=axis, dim=dim, keepdims=keepdims)


@torch_fn
def q50(x, axis=-1, dim=None, keepdims=True):
    return quantile(x, 50, axis=axis, dim=dim, keepdims=keepdims)


@torch_fn
def q75(x, axis=-1, dim=None, keepdims=True):
    return quantile(x, 75, axis=axis, dim=dim, keepdims=keepdims)


@torch_fn
def nanmax(x, axis=-1, dim=None, keepdims=True):
    min_value = torch.finfo(x.dtype).min
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        for d in sorted(dim, reverse=True):
            x = x.nan_to_num(min_value).max(dim=d, keepdim=keepdims)[0]
    else:
        x = x.nan_to_num(min_value).max(dim=dim, keepdim=keepdims)[0]
    return x


@torch_fn
def nanmin(x, axis=-1, dim=None, keepdims=True):
    max_value = torch.finfo(x.dtype).max
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        for d in sorted(dim, reverse=True):
            x = x.nan_to_num(max_value).min(dim=d, keepdim=keepdims)[0]
    else:
        x = x.nan_to_num(max_value).min(dim=dim, keepdim=keepdims)[0]
    return x


@torch_fn
def nanmean(x, axis=-1, dim=None, keepdims=True):
    return torch.nanmean(x, dim=dim, keepdim=keepdims)


# @torch_fn
# def nanvar(x, axis=-1, dim=None, keepdims=True):
#     tensor_mean = x.nanmean(dim=dim, keepdim=True)
#     return (x - tensor_mean).square().nanmean(dim=dim, keepdim=keepdims)


@torch_fn
def nanvar(x, axis=-1, dim=None, keepdims=True):
    tensor_mean = nanmean(x, dim=dim, keepdims=True)
    return (x - tensor_mean).square().nanmean(dim=dim, keepdim=keepdims)


@torch_fn
def nanstd(x, axis=-1, dim=None, keepdims=True):
    return torch.sqrt(nanvar(x, dim=dim, keepdims=keepdims))


@torch_fn
def nanzscore(x, axis=-1, dim=None, keepdims=True):
    _mean = nanmean(x, dim=dim, keepdims=keepdims)
    _std = nanstd(x, dim=dim, keepdims=keepdims)
    return (x - _mean) / _std


@torch_fn
def nankurtosis(x, axis=-1, dim=None, keepdims=True):
    zscores = nanzscore(x, axis=axis, keepdims=keepdims)
    return (
        torch.nanmean(torch.pow(zscores, 4.0), dim=dim, keepdim=keepdims) - 3.0
    )


@torch_fn
def nanskewness(x, axis=-1, dim=None, keepdims=True):
    zscores = nanzscore(x, axis=axis, keepdims=keepdims)
    return torch.nanmean(torch.pow(zscores, 3.0), dim=dim, keepdim=keepdims)


@torch_fn
def nanprod(x, axis=-1, dim=None, keepdims=True):
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        for d in sorted(dim, reverse=True):
            x = x.nan_to_num(1).prod(dim=d, keepdim=keepdims)
    else:
        x = x.nan_to_num(1).prod(dim=dim, keepdim=keepdims)
    return x


@torch_fn
def nancumprod(x, axis=-1, dim=None, keepdims=True):
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        raise ValueError("cumprod does not support multiple dimensions")
    return x.nan_to_num(1).cumprod(dim=dim)


@torch_fn
def nancumsum(x, axis=-1, dim=None, keepdims=True):
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        raise ValueError("cumsum does not support multiple dimensions")
    return x.nan_to_num(0).cumsum(dim=dim)


@torch_fn
def nanargmin(x, axis=-1, dim=None, keepdims=True):
    max_value = torch.finfo(x.dtype).max
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        for d in sorted(dim, reverse=True):
            x = x.nan_to_num(max_value).argmin(dim=d, keepdim=keepdims)
    else:
        x = x.nan_to_num(max_value).argmin(dim=dim, keepdim=keepdims)
    return x


@torch_fn
def nanargmax(x, axis=-1, dim=None, keepdims=True):
    min_value = torch.finfo(x.dtype).min
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        for d in sorted(dim, reverse=True):
            x = x.nan_to_num(min_value).argmax(dim=d, keepdim=keepdims)
    else:
        x = x.nan_to_num(min_value).argmax(dim=dim, keepdim=keepdims)
    return x


@torch_fn
def nanquantile(x, q, axis=-1, dim=None, keepdims=True):
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        for d in sorted(dim, reverse=True):
            mask = ~torch.isnan(x)
            x_filtered = torch.where(mask, x, torch.tensor(float("inf")))
            x = torch.quantile(x_filtered, q / 100, dim=d, keepdim=keepdims)
    else:
        mask = ~torch.isnan(x)
        x_filtered = torch.where(mask, x, torch.tensor(float("inf")))
        x = torch.quantile(x_filtered, q / 100, dim=dim, keepdim=keepdims)
    return x


@torch_fn
def nanq25(x, axis=-1, dim=None, keepdims=True):
    return nanquantile(x, 25, axis=axis, dim=dim, keepdims=keepdims)


@torch_fn
def nanq50(x, axis=-1, dim=None, keepdims=True):
    return nanquantile(x, 50, axis=axis, dim=dim, keepdims=keepdims)


@torch_fn
def nanq75(x, axis=-1, dim=None, keepdims=True):
    return nanquantile(x, 75, axis=axis, dim=dim, keepdims=keepdims)


if __name__ == "__main__":
    # from mngs.stats.descriptive import *

    x = np.random.rand(4, 3, 2)
    print(describe(x, dim=(1, 2), keepdims=True)[0].shape)
    print(describe(x, funcs="all", dim=(1, 2), keepdims=True)[0].shape)

# EOF
