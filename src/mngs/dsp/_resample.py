#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-04 20:02:10 (ywatanabe)"


import torchaudio.transforms as T
from mngs.general import torch_fn


@torch_fn
def resample(x, src_fs, tgt_fs):
    resampler = T.Resample(src_fs, tgt_fs, dtype=x.dtype).to(x.device)
    return resampler(x)
