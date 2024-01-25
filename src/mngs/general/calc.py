#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-25 15:41:31 (ywatanabe)"

import mngs
import numpy as np
import torch


def to_z(x, axis):
    if isinstance(x, torch.Tensor):  # [REVISED]
        return (x - x.mean(dim=axis, keepdim=True)) / x.std(
            dim=axis, keepdim=True
        )  # [REVISED]
    if isinstance(x, np.ndarray):  # [REVISED]
        return (x - x.mean(axis=axis, keepdims=True)) / x.std(
            axis=axis, keepdims=True
        )


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
