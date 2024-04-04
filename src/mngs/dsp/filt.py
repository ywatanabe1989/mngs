#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-04 13:23:08 (ywatanabe)"


import mngs
import scipy.ndimage
import torch
from mngs.general import torch_fn
from mngs.nn import BandPassFilter, GaussianFilter

# @torch_fn
# def resample(x, src_fs, tgt_fs):
#     resampler = T.Resample(src_fs, tgt_fs, dtype=x.dtype).to(x.device)
#     return resampler(x)


@torch_fn
def gauss(x, sigma):
    return GaussianFilter(sigma)(x)


@torch_fn
def bandpass(x, samp_rate, low_hz, high_hz):
    return BandPassFilter(low_hz, high_hz, samp_rate)(x)


@torch_fn
def bandstop(x, samp_rate, low_hz, high_hz):
    return BandPassFilter(low_hz, high_hz, samp_rate)(x)


def _custom_print(x):
    print(type(x), x.shape)


if __name__ == "__main__":
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
        _custom_print(gauss(xx, 6))
        _custom_print(bandpass(xx, src_fs, low_hz=55, high_hz=65))
        _custom_print(bandstop(xx, src_fs, low_hz=55, high_hz=65))
