#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-10 10:14:03 (ywatanabe)"

import torch
import torch.nn.functional as F


def _zero_pad_1d(x, target_length):
    padding_needed = target_length - x.size(0)
    padding_left = padding_needed // 2
    padding_right = padding_needed - padding_left
    return F.pad(x, (padding_left, padding_right), "constant", 0)


def zero_pad(xs, dim=0):
    max_len = max([x.size(0) for x in xs])
    return torch.stack([_zero_pad_1d(x, max_len) for x in xs], dim=dim)
