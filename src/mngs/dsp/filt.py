#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-04 21:07:15 (ywatanabe)"


import mngs
from mngs.general import torch_fn
from mngs.nn import BandPassFilter, BandStopFilter, GaussianFilter


@torch_fn
def gauss(x, sigma):
    return GaussianFilter(sigma)(x)


@torch_fn
def bandpass(x, samp_rate, low_hz, high_hz):
    return BandPassFilter(low_hz, high_hz, samp_rate)(x)


@torch_fn
def bandstop(x, samp_rate, low_hz, high_hz):
    return BandStopFilter(low_hz, high_hz, samp_rate)(x)


def _custom_print(x):
    print(type(x), x.shape)


if __name__ == "__main__":
    import torch

    t_sec = 10
    src_fs = 1024
    tgt_fs = 128
    freqs_hz = [30, 60, 100, 200, 1000]

    xx, tt, fs = mngs.dsp.demo_sig(
        t_sec=t_sec, fs=src_fs, freqs_hz=freqs_hz, type="ripple"
    )
    tt_resampled = mngs.dsp.resample(tt, src_fs, tgt_fs)

    # Filtering
    filted_bp = mngs.dsp.filt.bandpass(xx, fs, low_hz=20, high_hz=50)
    filted_bs = mngs.dsp.filt.bandstop(xx, fs, low_hz=20, high_hz=50)
    filted_gauss = mngs.dsp.filt.gauss(xx, sigma=3)

    # Plots
    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=True)
    i_batch = 0
    i_ch = 0
    axes[0].plot(tt, xx[i_batch, i_ch], label="Original")
    axes[1].plot(tt_resampled, resampled[i_batch, i_ch], label="Resampled")
    axes[2].plot(tt, filted_bp[i_batch, i_ch], label="Bandpass-filtered")
    # axes[3].plot(tt, filted_bs[i_batch, i_ch], label="Bandstop-filtered")
    # axes[4].plot(tt, filted_gauss[i_batch, i_ch], label="Gaussian-filtered")
    for ax in axes:
        ax.legend(loc="upper left")
    plt.show()

    mngs.dsp.filt.bandpass(x, src_fs, low_hz=55, high_hz=65)

    _custom_print(x)
    for xx in [x, torch.tensor(x)]:
        _custom_print(gauss(xx, 6))
        _custom_print(bandpass(xx, src_fs, low_hz=55, high_hz=65))
        _custom_print(bandstop(xx, src_fs, low_hz=55, high_hz=65))
