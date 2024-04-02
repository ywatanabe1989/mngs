#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-31 16:17:40 (ywatanabe)"

import scipy


def down_sample_1d(x, src_fs, tgt_fs):
    factor = int(src_fs / tgt_fs)
    assert factor == int(factor)
    return scipy.signal.decimate(x, factor)
