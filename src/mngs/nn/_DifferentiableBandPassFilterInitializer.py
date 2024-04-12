#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-12 16:54:52 (ywatanabe)"

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
from torchaudio.prototype.functional import sinc_impulse_response


# Functions
class DifferentiableBandPassFilterInitializer(nn.Module):
    def __init__(
        self,
        sig_len,
        fs,
        pha_low_hz=2,
        pha_high_hz=20,
        pha_n_bands=30,
        amp_low_hz=60,
        amp_high_hz=160,
        amp_n_bands=50,
        cycle=3,
    ):
        """
        Initializes a differentiable FIR filter design.
        The parameters low_hz and high_hz can be learned during training.
        """
        super().__init__()

        # Learnable parameters
        self.pha_mids = nn.Parameter(
            torch.linspace(pha_low_hz, pha_high_hz, pha_n_bands)
        )
        self.amp_mids = nn.Parameter(
            torch.linspace(amp_low_hz, amp_high_hz, amp_n_bands)
        )

        # Define cutoff frequencies as learnable parameters
        pha_lows, pha_highs = self.define_freqs(self.pha_mids, factor=4.0)
        amp_lows, amp_highs = self.define_freqs(self.amp_mids, factor=8.0)

        # Define orders
        pha_orders = self.define_orders(pha_lows, fs, sig_len, cycle)
        amp_orders = self.define_orders(amp_lows, fs, sig_len, cycle)
        max_order = max(pha_orders + amp_orders)

        # Initialize filters
        pha_bp_filters = self.calc_filters(pha_lows, pha_highs, fs, max_order)
        amp_bp_filters = self.calc_filters(amp_lows, amp_highs, fs, max_order)
        self.filters = torch.vstack([pha_bp_filters, amp_bp_filters])

    @staticmethod
    def define_freqs(mids, factor):
        lows = mids - mids / factor
        highs = mids + mids / factor
        return lows, highs

    @staticmethod
    def define_orders(lows_hz, fs, sig_len, cycle):
        def define_order(low_hz, fs, sig_len, cycle):
            order = cycle * int((fs // low_hz))
            order = order if 3 * order >= sig_len else (sig_len - 1) // 3
            order = mngs.gen.to_even(order)
            return order

        return [define_order(ll.item(), fs, sig_len, cycle) for ll in lows_hz]

    @staticmethod
    def define_window(order):
        n = torch.arange(order)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (order - 1))
        return window

    @staticmethod
    def calc_filters(lows_hz, highs_hz, fs, order):
        nyq = fs / 2.0
        order = mngs.gen.to_odd(order)
        # lowpass filters
        irs_ll = sinc_impulse_response(lows_hz / nyq, window_size=order)
        irs_hh = sinc_impulse_response(highs_hz / nyq, window_size=order)
        irs = irs_ll - irs_hh
        return irs

    # def concatenate_filters(self, pha_filters, amp_filters):
    #     # Max order
    #     max_order = 0
    #     for ff in pha_filters + amp_filters:
    #         max_order = max(max_order, len(ff))

    #     # Zero-pad filters symmetrically to the length of the longest filter
    #     concatenated_filters = []
    #     for filt in pha_filters + amp_filters:
    #         padding_size = max_order - len(filt)
    #         zero_padding_before = torch.zeros(padding_size // 2)
    #         zero_padding_after = torch.zeros((padding_size + 1) // 2)
    #         concatenated_filter = torch.cat(
    #             [zero_padding_before, filt, zero_padding_after]
    #         )
    #         concatenated_filters.append(concatenated_filter)
    #     return torch.stack(concatenated_filters)

    def forward(self):
        return self.filters, self.pha_mids, self.amp_mids


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
    kernels.sum().backward()  # OK. The filtering bands are trainable with backpropagation.

    mngs.dsp.utils.filter.plot_filter_responses(
        kernels[0].detach().cpu().numpy(), fs
    )
    # plt.show()

# EOF

"""
/home/ywatanabe/proj/entrance/mngs/nn/_DifferentiableBandPassFilterInitializer.py
"""
