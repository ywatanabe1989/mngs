#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 00:24:36 (ywatanabe)"
# File: ./mngs_repo/src/mngs/gen/_norm.py

import torch

from ..decorators import torch_fn
from ..torch import nanstd


@torch_fn
def to_z(x, dim=-1, device="cuda"):
    return (x - x.mean(dim=dim, keepdim=True)) / x.std(dim=dim, keepdim=True)

@torch_fn
def to_nanz(x, dim=-1, device="cuda"):
    nan_mean = torch.nanmean(x, dim=dim, keepdim=True)
    nan_std = nanstd(x, dim=dim, keepdim=True)
    return (x - nan_mean) / nan_std


@torch_fn
def to_1_1(x, amp=1.0, dim=-1, fn="mean", device="cuda"):
    MM = x.max(dim=dim, keepdims=True)[0].abs()
    mm = x.min(dim=dim, keepdims=True)[0].abs()
    return amp * x / torch.maximum(MM, mm)


@torch_fn
def unbias(x, dim=-1, fn="mean", device="cuda"):
    if fn == "mean":
        return x - x.mean(dim=dim, keepdims=True)
    if fn == "min":
        return x - x.min(dim=dim, keepdims=True)[0]


# def to_z(x, axis):
#     if torch.Tensor:
#         return (x - x.mean(dim=axis, keepdims=True)) / x.std(dim=axis, keepdims=True)
#     if np.ndarray:
#         return (x - x.mean(axis=axis, keepdims=True)) / x.std(axis=axis, keepdims=True)

#     x, x_type = mngs.gen.my2array(x)

#     dtype_orig = x.dtype
#     x = x.astype(np.float64)
#     z = (x - x.mean(axis=axis, keepdims=True)) / x.std(axis=axis, keepdims=True)
#     z = z.astype(dtype_orig)

#     if x_type == "tensor":
#         return torch.tensor(z)
#     if x_type == "numpy":
#         return z


# EOF
