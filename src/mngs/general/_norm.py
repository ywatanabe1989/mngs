#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-02 21:34:33 (ywatanabe)"

import mngs
import numpy as np
import torch
from mngs.general import torch_fn


def to_z(x, axis):
    if isinstance(x, torch.Tensor):  # [REVISED]
        return (x - x.mean(dim=axis, keepdim=True)) / x.std(
            dim=axis, keepdim=True
        )  # [REVISED]
    if isinstance(x, np.ndarray):  # [REVISED]
        return (x - x.mean(axis=axis, keepdims=True)) / x.std(
            axis=axis, keepdims=True
        )


@torch_fn
def to_z(x, dim=-1):
    return (x - x.mean(dim=dim, keepdim=True)) / x.std(dim=dim, keepdim=True)


@torch_fn
def to_1_1(x, amp=1.0, dim=-1, fn="mean"):
    MM = x.max(dim=dim, keepdims=True)[0].abs()
    mm = x.min(dim=dim, keepdims=True)[0].abs()
    return amp * x / torch.maximum(MM, mm)


@torch_fn
def unbias(x, dim=-1, fn="mean"):
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
