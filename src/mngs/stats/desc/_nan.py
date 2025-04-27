#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-25 20:51:05 (ywatanabe)"
# File: ./mngs_repo/src/mngs/stats/desc/_nan.py

THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/stats/desc/_nan.py"

from mngs.decorators import torch_fn, batch_fn
import torch

@torch_fn
@batch_fn
def nanmax(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    min_value = torch.finfo(x.dtype).min
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        for d in sorted(dim, reverse=True):
            x = x.nan_to_num(min_value).max(dim=d, keepdims=keepdims)[0]
    else:
        x = x.nan_to_num(min_value).max(dim=dim, keepdims=keepdims)[0]
    return x

@torch_fn
@batch_fn
def nanmin(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    max_value = torch.finfo(x.dtype).max
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        for d in sorted(dim, reverse=True):
            x = x.nan_to_num(max_value).min(dim=d, keepdims=keepdims)[0]
    else:
        x = x.nan_to_num(max_value).min(dim=dim, keepdims=keepdims)[0]
    return x

@torch_fn
@batch_fn
def nansum(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    return torch.nansum(x, dim=dim, keepdims=keepdims)

@torch_fn
@batch_fn
def nanmean(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    return torch.nanmean(x, dim=dim, keepdims=keepdims)

@torch_fn
@batch_fn
def nanvar(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    tensor_mean = nanmean(x, dim=dim, keepdims=True)
    return (x - tensor_mean).square().nanmean(dim=dim, keepdims=keepdims)

@torch_fn
@batch_fn
def nanstd(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    return torch.sqrt(nanvar(x, dim=dim, keepdims=keepdims))


# @torch_fn
# def nanzscore(x, axis=-1, dim=None, batch_size=None, keepdims=True):
#     _mean = nanmean(x, dim=dim, keepdims=True)
#     _std = nanstd(x, dim=dim, keepdims=True)
#     zscores = (x - _mean) / _std
#     return zscores if keepdims else zscores.squeeze(dim)
@torch_fn
@batch_fn
def nanzscore(x, axis=-1, dim=None, batch_size=None, keepdims=True):
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        _mean = nanmean(x, dim=dim, keepdims=True)
        _std = nanstd(x, dim=dim, keepdims=True)
    else:
        _mean = nanmean(x, dim=dim, keepdims=True)
        _std = nanstd(x, dim=dim, keepdims=True)
    zscores = (x - _mean) / _std
    return zscores if keepdims else zscores.squeeze(dim)


# @torch_fn
# def nankurtosis(x, axis=-1, dim=None, batch_size=None, keepdims=False):
#     zscores = nanzscore(x, axis=axis, keepdims=True)
#     n = (~torch.isnan(x)).sum(dim=dim, keepdim=True).to(x.dtype)  # Changed this line
#     k = torch.nanmean(torch.pow(zscores, 4.0), dim=dim, keepdims=keepdims)
#     correction = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
#     return correction * k - 3 * (n - 1)**2 / ((n - 2) * (n - 3))


# @torch_fn
# def nankurtosis(x, axis=-1, dim=None, batch_size=None, keepdims=False):
#     dim = axis if dim is None else dim
#     if isinstance(dim, (tuple, list)):
#         zscores = nanzscore(x, dim=dim, keepdims=True)
#         n = (~torch.isnan(x)).sum(dim=dim, keepdim=True).to(x.dtype)
#         k = torch.nanmean(torch.pow(zscores, 4.0), dim=dim, keepdims=keepdims)
#         correction = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
#         result = correction * k - 3 * (n - 1)**2 / ((n - 2) * (n - 3))
#         return result.squeeze() if not keepdims else result
#     else:
#         # Original code for single dimension
#         zscores = nanzscore(x, dim=dim, keepdims=True)
#         n = (~torch.isnan(x)).sum(dim=dim, keepdim=True).to(x.dtype)
#         k = torch.nanmean(torch.pow(zscores, 4.0), dim=dim, keepdims=keepdims)
#         correction = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
#         result = correction * k - 3 * (n - 1)**2 / ((n - 2) * (n - 3))
#         return result.squeeze() if not keepdims else result

# @torch_fn
# def nanskewness(x, axis=-1, dim=None, batch_size=None, keepdims=False):
#     zscores = nanzscore(x, axis=axis, keepdims=True)
#     n = (~torch.isnan(x)).sum(dim=dim, keepdim=True).to(x.dtype)  # Changed this line
#     s = torch.nanmean(torch.pow(zscores, 3.0), dim=dim, keepdims=keepdims)
#     correction = n**2 / ((n - 1) * (n - 2))
#     return correction * s

@torch_fn
@batch_fn
def nankurtosis(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    zscores = nanzscore(x, axis=axis, keepdims=True)
    return (
        torch.nanmean(torch.pow(zscores, 4.0), dim=dim, keepdims=keepdims)
        - 3.0
    )


@torch_fn
@batch_fn
def nanskewness(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    zscores = nanzscore(x, axis=axis, keepdims=True)
    return torch.nanmean(torch.pow(zscores, 3.0), dim=dim, keepdims=keepdims)


@torch_fn
@batch_fn
def nanprod(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        for d in sorted(dim, reverse=True):
            x = x.nan_to_num(1).prod(dim=d, keepdims=keepdims)
    else:
        x = x.nan_to_num(1).prod(dim=dim, keepdims=keepdims)
    return x


@torch_fn
@batch_fn
def nancumprod(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        raise ValueError("cumprod does not support multiple dimensions")
    return x.nan_to_num(1).cumprod(dim=dim)


@torch_fn
@batch_fn
def nancumsum(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        raise ValueError("cumsum does not support multiple dimensions")
    return x.nan_to_num(0).cumsum(dim=dim)


@torch_fn
@batch_fn
def nanargmin(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    max_value = torch.finfo(x.dtype).max
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        for d in sorted(dim, reverse=True):
            x = x.nan_to_num(max_value).argmin(dim=d, keepdims=keepdims)
    else:
        x = x.nan_to_num(max_value).argmin(dim=dim, keepdims=keepdims)
    return x


@torch_fn
@batch_fn
def nanargmax(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    min_value = torch.finfo(x.dtype).min
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        for d in sorted(dim, reverse=True):
            x = x.nan_to_num(min_value).argmax(dim=d, keepdims=keepdims)
    else:
        x = x.nan_to_num(min_value).argmax(dim=dim, keepdims=keepdims)
    return x


@torch_fn
@batch_fn
def nanquantile(x, q, axis=-1, dim=None, batch_size=None, keepdims=False):
    dim = axis if dim is None else dim
    if isinstance(dim, (tuple, list)):
        for d in sorted(dim, reverse=True):
            mask = ~torch.isnan(x)
            x_filtered = torch.where(mask, x, torch.tensor(float("inf")))
            x = torch.quantile(x_filtered, q / 100, dim=d, keepdims=keepdims)
    else:
        mask = ~torch.isnan(x)
        x_filtered = torch.where(mask, x, torch.tensor(float("inf")))
        x = torch.quantile(x_filtered, q / 100, dim=dim, keepdims=keepdims)
    return x


@torch_fn
@batch_fn
def nanq25(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    return nanquantile(x, 25, axis=axis, dim=dim, keepdims=keepdims)


@torch_fn
@batch_fn
def nanq50(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    return nanquantile(x, 50, axis=axis, dim=dim, keepdims=keepdims)


@torch_fn
@batch_fn
def nanq75(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    return nanquantile(x, 75, axis=axis, dim=dim, keepdims=keepdims)


@torch_fn
@batch_fn
def nancount(x, axis=-1, dim=None, batch_size=None, keepdims=False):
    """Count number of non-NaN values along specified dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    axis : int, default=-1
        Deprecated. Use dim instead
    dim : int or tuple of ints, optional
        Dimension(s) along which to count
    keepdims : bool, default=True
        Whether to keep reduced dimensions

    Returns
    -------
    torch.Tensor
        Count of non-NaN values
    """
    dim = axis if dim is None else dim
    mask = ~torch.isnan(x)

    if isinstance(dim, (tuple, list)):
        for d in sorted(dim, reverse=True):
            mask = mask.sum(dim=d, keepdims=keepdims)
    else:
        mask = mask.sum(dim=dim, keepdims=keepdims)

    return mask


# EOF
