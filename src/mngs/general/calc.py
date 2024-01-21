#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-21 19:00:33 (ywatanabe)"

import mngs
import numpy as np
import torch

def to_z(x, axis):
    x, x_type = mngs.gen.my2array(x)
    dtype_orig = x.dtype
    x = x.astype(np.float64)
    z = (x - x.mean(axis=axis, keepdims=True)) / x.std(axis=axis, keepdims=True)
    z = z.astype(dtype_orig)

    if x_type == "tensor":
        return torch.tensor(z)
    if x_type == "numpy":
        return z

