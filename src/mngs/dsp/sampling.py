#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-20 14:20:50 (ywatanabe)"#!/usr/bin/env python3

import random

import numpy as np
import scipy


def down_sample_1d(x, src_fs, tgt_fs):
    factor = int(src_fs / tgt_fs)
    assert factor == int(factor)
    return scipy.signal.decimate(x, factor)


def random_pts(max_sec, samp_rate, start_sec=None, dur_sec=10):

    start_sec = (
        random.randint(0, max_sec) if (start_sec is None) else start_sec
    )
    end_sec = start_sec + dur_sec

    start_pts = int(start_sec * samp_rate)
    end_pts = int(end_sec * samp_rate)

    x_time_sec = np.linspace(0, int(dur_sec), int(dur_sec) * samp_rate)

    return x_time_sec, start_sec, end_sec, start_pts, end_pts
