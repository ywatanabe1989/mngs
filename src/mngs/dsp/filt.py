#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-03 13:19:54 (ywatanabe)"


import time
from functools import partial

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from mngs.general import numpy_fn, torch_fn
from mngs.nn import BandPassFilter, BandStopFilter, GaussianFilter
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt
from scipy.signal import butter, sosfilt, sosfreqz


@torch_fn
def resample(x, src_fs, tgt_fs, cuda=True):
    resampler = T.Resample(src_fs, tgt_fs, dtype=x.dtype).to(x.device)
    return resampler(x)


@torch_fn
def gauss(x, sigma, cuda=True):
    return GaussianFilter(sigma)(x)


@torch_fn
def bandpass(x, samp_rate, low_hz, high_hz, cuda=True):
    return BandPassFilter(low_hz, high_hz, samp_rate)(x)


@torch_fn
def bandstop(x, samp_rate, low_hz, high_hz, cuda=True):
    return BandPassFilter(low_hz, high_hz, samp_rate)(x)


def _custom_print(x):
    print(type(x), x.shape)


if __name__ == "__main__":
    import seaborn as sns

    t_sec = 10
    src_fs = 1024
    tgt_fs = 128
    freqs_hz = [30, 60, 100, 200, 1000]

    x = mngs.dsp.demo_sig(
        t_sec=t_sec, fs=src_fs, freqs_hz=freqs_hz, type="ripple"
    )

    mngs.dsp.filt.bandpass(x, src_fs, low_hz=55, high_hz=65)

    _custom_print(x)
    for xx in [x, torch.tensor(x)]:
        # _custom_print(resample(xx, src_fs, tgt_fs))
        _custom_print(gauss(xx, 6))
        _custom_print(bandpass(xx, src_fs, low_hz=55, high_hz=65))
        _custom_print(bandstop(xx, src_fs, low_hz=55, high_hz=65))
        # _custom_print(wavelet(x, src_fs, num_scales=16, cuda=True))
