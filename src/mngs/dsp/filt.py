#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-10 15:55:58 (ywatanabe)"


import mngs
import numpy as np
from mngs.general import torch_fn
from mngs.nn import (
    BandPassFilter,
    BandStopFilter,
    GaussianFilter,
    HighPassFilter,
    LowPassFilter,
)


@torch_fn
def gauss(x, sigma, t=None):
    return GaussianFilter(sigma)(x, t=t)


@torch_fn
def bandpass(x, fs, bands, t=None):
    return BandPassFilter(bands, fs, x.shape)(x, t=t)


@torch_fn
def bandstop(x, fs, bands, t=None):
    return BandStopFilter(bands, fs, x.shape)(x, t=t)


@torch_fn
def lowpass(x, fs, cutoffs_hz, t=None):
    return LowPassFilter(cutoffs_hz, fs, x.shape)(x, t=t)


@torch_fn
def highpass(x, fs, cutoffs_hz, t=None):
    return HighPassFilter(cutoffs_hz, fs, x.shape)(x, t=t)


def _custom_print(x):
    print(type(x), x.shape)


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import mngs
    import torch

    # Start
    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(sys, plt)

    # Parametes
    T_SEC = 10
    SRC_FS = 1024
    TGT_FS = 128
    FREQS_HZ = [30, 60, 100, 200, 1000]
    SIG_TYPE = "periodic"
    BANDS = np.vstack([[45, 75]])

    # Demo Signal
    xx, tt, fs = mngs.dsp.demo_sig(
        t_sec=T_SEC,
        fs=SRC_FS,
        freqs_hz=FREQS_HZ,
        sig_type=SIG_TYPE,
    )

    # Resampling
    x_r, t_r = mngs.dsp.resample(xx, fs, TGT_FS, t=tt)

    # Filtering
    x_bp, t_bp = mngs.dsp.filt.bandpass(xx, fs, BANDS, t=tt)
    x_bs, t_bs = mngs.dsp.filt.bandstop(xx, fs, BANDS, t=tt)
    x_lp, t_lp = mngs.dsp.filt.lowpass(xx, fs, BANDS[:, 0], t=tt)
    x_hp, t_hp = mngs.dsp.filt.highpass(xx, fs, BANDS[:, 1], t=tt)
    x_g, t_g = mngs.dsp.filt.gauss(xx, sigma=3, t=tt)
    filted = {
        "Original": (xx, tt, fs),
        "Resampled": (x_r, t_r, TGT_FS),
        "Bandpass-filtered": (x_bp, t_bp, fs),
        "Bandstop-filtered": (x_bs, t_bs, fs),
        "Lowpass-filtered": (x_lp, t_lp, fs),
        "Highpass-filtered": (x_hp, t_hp, fs),
        "Gaussian-filtered": (x_g, t_g, fs),
    }

    # Plots traces
    fig, axes = plt.subplots(
        nrows=len(filted), ncols=1, sharex=True, sharey=True
    )
    i_batch = 0
    i_ch = 0
    i_filt = 0
    for ax, (k, v) in zip(axes, filted.items()):
        _xx, _tt, _fs = v
        if _xx.ndim == 3:
            _xx = _xx[i_batch, i_ch]
        elif _xx.ndim == 4:
            _xx = _xx[i_batch, i_ch, i_filt]
        ax.plot(_tt, _xx, label=k)
        ax.legend(loc="upper left")
    fig.supxlabel("Time [s]")
    fig.supylabel("Amplitude")
    mngs.io.save(fig, CONFIG["SDIR"] + "traces.png")

    # Calculates and Plots PSD
    fig, axes = plt.subplots(
        nrows=len(filted), ncols=1, sharex=True, sharey=True
    )
    i_batch = 0
    i_ch = 0
    i_filt = 0
    for ax, (k, v) in zip(axes, filted.items()):
        _xx, _tt, _fs = v

        _psd, ff = mngs.dsp.psd(_xx, _fs)
        if _psd.ndim == 3:
            _psd = _psd[i_batch, i_ch]
        elif _psd.ndim == 4:
            _psd = _psd[i_batch, i_ch, i_filt]

        ax.plot(ff, _psd, label=k)
        ax.legend(loc="upper left")
    fig.supxlabel("Frequency [Hz]")
    fig.supylabel("Power Spectral Density")
    mngs.io.save(fig, CONFIG["SDIR"] + "psd.png")

    # Close
    mngs.gen.close(CONFIG)

    """
    /home/ywatanabe/proj/mngs/src/mngs/dsp/filt.py
    """

    # EOF
