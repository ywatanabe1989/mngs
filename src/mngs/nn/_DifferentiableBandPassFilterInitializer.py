#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-11 15:10:45 (ywatanabe)"

"""
This script does XYZ.
"""


import math

# Imports
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Config
CONFIG = mngs.gen.load_configs()


# Functions
class DifferentiableBandPassFilterInitializer(nn.Module):
    def __init__(
        self,
        sig_len,
        fs,
        pha_low_hz=2,
        pha_high_hz=20,
        pha_n_bands=30,
        amp_low_hz=80,
        amp_high_hz=160,
        amp_n_bands=50,
        cycle=3,
    ):
        """
        Initializes a differentiable FIR filter design.
        The parameters low_hz and high_hz can be learned during training.
        """
        super().__init__()

        # Attributes
        self.fs = fs

        # Define cutoff frequencies as learnable parameters
        (
            self.pha_mids,
            pha_lows,
            pha_highs,
        ) = self.define_freqs(pha_low_hz, pha_high_hz, pha_n_bands, factor=4.0)
        (
            self.amp_mids,
            amp_lows,
            amp_highs,
        ) = self.define_freqs(amp_low_hz, amp_high_hz, amp_n_bands, factor=8.0)

        # Order
        pha_order = self.define_order(pha_low_hz, fs, sig_len, cycle)
        amp_order = self.define_order(amp_low_hz, fs, sig_len, cycle)
        order = max(pha_order, amp_order)

        # Define the window for the filter (e.g., Hamming window)
        window = self.define_window(order)

        # Construct filters
        pha_bp_filters = self.calc_filters(
            order, pha_lows, pha_highs, fs, window
        )
        amp_bp_filters = self.calc_filters(
            order, amp_lows, amp_highs, fs, window
        )

        self.filters = torch.vstack([pha_bp_filters, amp_bp_filters])

    def forward(
        self,
    ):
        return self.filters, self.pha_mids, self.amp_mids

    @staticmethod
    def define_freqs(low_hz, high_hz, n_bands, factor):
        mids = nn.Parameter(torch.linspace(low_hz, high_hz, n_bands))
        lows = mids - mids / factor
        highs = mids + mids / factor
        return mids, lows, highs

    @staticmethod
    def define_order(low_hz, fs, sig_len, cycle):
        order = cycle * int((fs // low_hz))
        if 3 * order < sig_len:
            order = (sig_len - 1) // 3
        order = mngs.gen.to_even(order)
        return order

    @staticmethod
    def define_window(order):
        n = torch.arange(order)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (order - 1))
        return window

    @staticmethod
    def calc_filters(order, lows, highs, fs, window):
        tt = torch.linspace(-order // 2, order // 2, order)
        sinc_low = torch.sinc(2 * lows.unsqueeze(1) / fs * tt.unsqueeze(0))
        sinc_high = torch.sinc(2 * highs.unsqueeze(1) / fs * tt.unsqueeze(0))
        bp_filters = (sinc_high - sinc_low) * window.unsqueeze(0)
        return bp_filters


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    # Demo signal
    freqs_hz = [10, 30, 100, 300]
    fs = 1024
    xx, tt, fs = mngs.dsp.demo_sig(fs=fs, freqs_hz=freqs_hz)

    # Main
    kernels, pha_bands, amp_bands = DifferentiableBandPassFilterInitializer(
        xx.shape[-1], fs
    )()


#     # Checks the PSDs
#     psd, ff = mngs.dsp.psd(xx, fs)
#     psd_f, ff_f = mngs.dsp.psd(xf, fs)

#     # Plots
#     matplotlib.use("TkAgg")
#     fig, axes = mngs.plt.subplots(nrows=2)
#     axes[0].plot(ff, psd[0, 0], label="Original")
#     axes[1].plot(ff_f, psd_f[0, 0].detach().numpy(), label="Filted")
#     for ax in axes:
#         ax.legend(loc="upper left")
#     plt.show()

#     # Close
#     mngs.gen.close(CONFIG)

# # EOF

# """
# /ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/dsp/utils/_DIfferentialFIRFilter.py
# """
