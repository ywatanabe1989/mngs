#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-03 00:17:07 (ywatanabe)"

import torch
import torch.nn as nn
from mngs.general import torch_fn
from mngs.nn import PSD


@torch_fn
def psd(x, fs, dim=-1, cuda=True):
    psd, freqs = PSD(fs, dim=dim)(x)
    return psd, freqs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mngs

    fs = 512
    freqs_hz = [30, 100, 200]
    x = mngs.dsp.demo_sig(fs=fs, freqs_hz=freqs_hz)  # (8, 19, 384)
    pp, ff = psd(x, fs)
    print(pp.shape)
    print(ff.shape)

    plt, CC = mngs.plt.configure_mpl(plt)
    fig, ax = mngs.plt.subplots()
    ax.plot(ff, pp[0, 0])
    plt.show()
