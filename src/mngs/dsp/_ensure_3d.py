#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-02 23:32:59 (ywatanabe)"

from mngs.general import torch_fn


@torch_fn
def ensure_3d(x):
    if x.ndim == 1:  # assumes (seq_len,)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 2:  # assumes (batch_siize, seq_len)
        x = x.unsqueeze(1)
    return x
