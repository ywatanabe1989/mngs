#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-11 20:16:23 (ywatanabe)"

"""
This script does XYZ.
"""

# Imports

import os
import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mngs.general import torch_fn
from mngs.nn import ModulationIndex


@torch_fn
def modulation_index(pha, amp, n_bins=18):
    """
    pha: (batch_size, n_chs, n_freqs_pha, n_segments, seq_len)
    amp: (batch_size, n_chs, n_freqs_amp, n_segments, seq_len)
    """
    return ModulationIndex(n_bins=n_bins)(pha, amp)


def calc_pac_with_tensorpac(xx, fs, t_sec):
    import tensorpac
    from tensorpac import Pac

    # Morlet's Wavelet Transfrmation
    p = tensorpac.Pac(f_pha="hres", f_amp="hres", dcomplex="wavelet")

    # Bandpass Filtering and Hilbert Transformation
    i_batch, i_ch = 0, 0
    phases = p.filter(
        fs, xx[i_batch, i_ch], ftype="phase", n_jobs=1
    )  # (50, 20, 2048)
    amplitudes = p.filter(
        fs, xx[i_batch, i_ch], ftype="amplitude", n_jobs=1
    )  # (50, 20, 2048)

    # Calculates xpac
    k = 2
    p.idpac = (k, 0, 0)

    xpac = p.fit(phases, amplitudes)  # (50, 50, 20)
    pac = xpac.mean(axis=-1)  # (50, 50)

    # ## Plot
    # fig, ax = plt.subplots()
    # ax = p.comodulogram(
    #     pac, title=p.method.replace(" (", f" ({k})\n("), cmap="viridis"
    # )
    # ax = mngs.plt.ax.set_n_ticks(ax)
    # import ipdb

    # ipdb.set_trace()
    freqs_amp = p.f_amp.mean(axis=-1)
    freqs_pha = p.f_pha.mean(axis=-1)

    return phases, amplitudes, freqs_pha, freqs_amp, pac
    # return phases and amplitudes for future use in my implementation
    # as the aim of this code is to confirm the calculation of Modulation Index only
    # without considering bandpass filtering and hilbert transformation.


@torch_fn
def _reshape(x, batch_size=1, n_chs=1):
    return (
        torch.tensor(x)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, n_chs, 1, 1, 1)
    ).float()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mngs


# Config
CONFIG = mngs.gen.load_configs()

# Functions
# Your awesome code here :)

if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    # Parameters
    FS = 128
    T_SEC = 5

    # Demo signal
    xx, tt, fs = mngs.dsp.demo_sig(fs=FS, t_sec=T_SEC, sig_type="tensorpac")
    # xx.shape: (8, 19, 20, 512)

    # Tensorpac
    pha, amp, freqs_pha, freqs_amp, pac = calc_pac_with_tensorpac(
        xx,
        fs,
        t_sec=T_SEC,
    )
    # mngs.io.save((pha, amp, freqs_pha, freqs_amp), "/tmp/out.pkl")
    # pha, amp, freqs_pha, freqs_amp = mngs.io.load("/tmp/out.pkl")

    # GPU calculation
    pha, amp = _reshape(pha), _reshape(amp)
    pac = mngs.dsp.modulation_index(pha, amp)

    ## Convert y-axis
    i_batch, i_ch = 0, 0

    fig, ax = mngs.plt.subplots()
    ax.imshow2d(
        pac[i_batch, i_ch],
        cbar_label="PAC values",
    )
    ax = mngs.plt.ax.set_ticks(
        ax, xticks=freqs_pha.astype(int), yticks=freqs_amp.astype(int)
    )
    ax = mngs.plt.ax.set_n_ticks(ax)
    ax.set_xlabel("Frequency for phase [Hz]")
    ax.set_ylabel("Frequency for amplitude [Hz]")
    ax.set_title("GPU calculation")
    mngs.io.save(fig, CONFIG["SDIR"] + "MI.png")
    # plt.show()

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
%s
"""
