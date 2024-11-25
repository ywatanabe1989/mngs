#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-25 23:34:44 (ywatanabe)"
# File: ./mngs_repo/src/mngs/dsp/utils/_zero_pad.py

__file__ = "/data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dsp/utils/_zero_pad.py"

import torch
import torch.nn.functional as F
from ...decorators import torch_fn


def _zero_pad_1d(x, target_length):
    padding_needed = target_length - len(x)
    padding_left = padding_needed // 2
    padding_right = padding_needed - padding_left
    return F.pad(x, (padding_left, padding_right), "constant", 0)


def zero_pad(xs, dim=0):
    max_len = max([len(x) for x in xs])
    return torch.stack([_zero_pad_1d(x, max_len) for x in xs], dim=dim)


# EOF
