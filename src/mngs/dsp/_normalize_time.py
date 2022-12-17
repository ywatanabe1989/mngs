#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-10-07 10:47:01 (ywatanabe)"

import torch
import numpy as np

def normalize_time(x):
    if type(x) == torch.Tensor:
        return (x - x.mean(dim=-1, keepdims=True)) \
            / x.std(dim=-1, keepdims=True)
    if type(x) == np.ndarray:
        return (x - x.mean(axis=-1, keepdims=True)) \
            / x.std(axis=-1, keepdims=True)

if __name__ == "__main__":
    x = 100 * np.random.rand(16, 160, 1000)
    print(_normalize_time(x))

    x = torch.randn(16, 160, 1000)
    print(_normalize_time(x))

    x = torch.randn(16, 160, 1000).cuda()
    print(_normalize_time(x))    
