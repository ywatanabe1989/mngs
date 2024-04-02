#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-02 20:27:51 (ywatanabe)"


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
def resample(x, src_fs, tgt_fs):
    """
    tgt_fs=512
    src_fs=128
    x = mngs.dsp.np.demo_sig(fs=tgt_fs)
    x = resample(x, tgt_fs, src_fs)
    """
    resampler = T.Resample(src_fs, tgt_fs, dtype=x.dtype).to(x.device)
    return resampler(x)


@torch_fn
def gauss(x, sigma, cuda=True):
    return GaussianFilter(sigma)(x)


@torch_fn
def bandpass(x, samp_rate, low_hz=55, high_hz=65, cuda=True):
    return BandPassFilter(low_hz, high_hz, samp_rate)(x)


@torch_fn
def bandstop(x, samp_rate, low_hz=55, high_hz=65, cuda=True):
    return BandPassFilter(low_hz, high_hz, samp_rate)(x)


def custom_print(x):
    print(type(x), x.shape)


if __name__ == "__main__":
    from mngs.dsp.feature_extractors import _rfft_1d  # fixme

    t_sec = 10
    src_fs = 1024
    tgt_fs = 128
    freqs_hz = [30, 60, 100, 200, 1000]

    def closest_power_base(N):
        # Calculate the nearest power of two upwards and downwards
        power_up = 2 ** np.ceil(np.log2(N))
        power_down = 2 ** np.floor(np.log2(N))

        # Determine which is closer to N
        if abs(power_up - N) < abs(power_down - N):
            return int(np.log2(power_up))
        else:
            return int(np.log2(power_down))

    x = mngs.dsp.np.demo_sig(
        t_sec=t_sec, fs=src_fs, freqs_hz=freqs_hz, type="ripple"
    )

    out = wavelet(x, src_fs, num_scales=closest_power_base(src_fs), cuda=True)

    import seaborn as sns

    sns.heatmap(out[0, 0])
    plt.show()

    custom_print(wavelet(x, src_fs, 3, cuda=False))

    custom_print(x)
    for xx in [x, torch.tensor(x)]:
        # custom_print(resample(xx, src_fs, tgt_fs))
        custom_print(gauss(xx, 6))
        custom_print(bandpass(xx, src_fs, low_hz=55, high_hz=65))
        custom_print(bandstop(xx, src_fs, low_hz=55, high_hz=65))
        custom_print(wavelet(x, src_fs, num_scales=16, cuda=True))
