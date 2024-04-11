#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-11 16:04:58 (ywatanabe)"

"""
This script does XYZ.
"""

import math
import os
import sys
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

# Imports
import mngs
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from mngs.general import torch_fn

# from mngs.nn import DifferentiableBandPassFilterInitializer
# from ._DifferentiableBandPassFilterInitializer import (
#     DifferentiableBandPassFilterInitializer,
# )


class BaseFilter1D(nn.Module):
    def __init__(self, fp16=False):
        super().__init__()
        self.fp16 = fp16

    @abstractmethod
    def init_kernels(
        self,
    ):
        """
        Abstract method to initialize filter kernels.
        Must be implemented by subclasses.
        """
        pass

    def forward(self, x, t=None, edge_len=0):
        """Apply the filter to input signal x with shape: (batch_size, n_chs, seq_len)"""

        # Shape check
        if self.fp16:
            x = x.half()

        x = mngs.dsp.ensure_3d(x)
        batch_size, n_chs, seq_len = x.shape

        # Kernel Check
        if self.kernels is None:
            raise ValueError("Filter kernels has not been initialized.")

        # Filtering
        x = self.flip_extend(x, self.kernel_size // 2)
        x = self.batch_conv(x, self.kernels, padding=0)
        x = x[..., :seq_len]  # fixme

        assert x.shape == (
            batch_size,
            n_chs,
            len(self.kernels),
            seq_len,
        ), f"The shape of the filtered signal ({x.shape}) does not match the expected shape: ({batch_size}, {n_chs}, {len(self.kernels)}, {seq_len})."

        # Edge remove
        x = self.remove_edges(x, edge_len)

        if t is None:
            return x
        else:
            t = self.remove_edges(t, edge_len)
            return x, t

    @property
    def kernel_size(
        self,
    ):
        ks = self.kernels.shape[-1]
        if not ks % 2 == 0:
            raise ValueError("Kernel size should be an even number.")
        return ks

    @staticmethod
    def flip_extend(x, extension_length):
        first_segment = x[:, :, :extension_length].flip(dims=[-1])
        last_segment = x[:, :, -extension_length:].flip(dims=[-1])
        return torch.cat([first_segment, x, last_segment], dim=-1)

    @staticmethod
    def batch_conv(x, kernels, padding="same"):
        """
        x: (batch_size, n_chs, seq_len)
        kernels: (n_kernels, seq_len_filt)
        """
        assert x.ndim == 3
        assert kernels.ndim == 2
        batch_size, n_chs, n_time = x.shape
        x = x.reshape(-1, x.shape[-1]).unsqueeze(1)
        kernels = kernels.unsqueeze(1)  # add the channel dimension
        n_kernels = len(kernels)
        filted = F.conv1d(x, kernels.type_as(x), padding=padding)
        return filted.reshape(batch_size, n_chs, n_kernels, -1)

    @staticmethod
    def remove_edges(x, edge_len):
        edge_len = x.shape[-1] // 8 if edge_len == "auto" else edge_len

        if 0 < edge_len:
            return x[..., edge_len:-edge_len]
        else:
            return x


class BandPassFilter(BaseFilter1D):
    def __init__(self, bands, fs, x_shape, fp16=False):
        super().__init__(fp16=fp16)

        self.fp16 = fp16

        # Ensures bands shape
        assert bands.ndim == 2

        # Check bands definitions
        nyq = fs / 2.0
        for ll, hh in bands:
            assert 0 < ll
            assert ll < hh
            assert hh < nyq

        # Prepare kernels
        kernels = self.init_kernels(x_shape[-1], fs, bands)
        if fp16:
            kernels = kernels.half()
        self.register_buffer(
            "kernels",
            kernels,
        )

    @staticmethod
    def init_kernels(seq_len, fs, bands):
        filters = [
            mngs.dsp.utils.design_filter(
                seq_len,
                fs,
                low_hz=ll,
                high_hz=hh,
                is_bandstop=False,
            )
            for ll, hh in bands
        ]

        kernels = mngs.dsp.utils.zero_pad(filters)
        kernels = mngs.dsp.utils.ensure_even_len(kernels)
        kernels = torch.tensor(kernels)
        return kernels


class BandStopFilter(BaseFilter1D):
    def __init__(self, bands, fs, x_shape):
        super().__init__()

        # Ensures bands shape
        assert bands.ndim == 2

        # Check bands definitions
        nyq = fs / 2.0
        for ll, hh in bands:
            assert 0 < ll
            assert ll < hh
            assert hh < nyq

        self.register_buffer(
            "kernels", self.init_kernels(x_shape[-1], fs, bands)
        )

    @staticmethod
    def init_kernels(seq_len, fs, bands):
        kernels = mngs.dsp.utils.zero_pad(
            [
                mngs.dsp.utils.design_filter(
                    seq_len, fs, low_hz=ll, high_hz=hh, is_bandstop=True
                )
                for ll, hh in bands
            ]
        )
        kernels = mngs.dsp.utils.ensure_even_len(kernels)
        return torch.tensor(kernels)


class LowPassFilter(BaseFilter1D):
    def __init__(self, cutoffs_hz, fs, x_shape):
        super().__init__()

        # Ensures bands shape
        assert cutoffs_hz.ndim == 1

        # Check bands definitions
        nyq = fs / 2.0
        for cc in cutoffs_hz:
            assert 0 < cc
            assert cc < nyq

        self.register_buffer(
            "kernels", self.init_kernels(x_shape[-1], fs, cutoffs_hz)
        )

    @staticmethod
    def init_kernels(seq_len, fs, cutoffs_hz):
        kernels = mngs.dsp.utils.zero_pad(
            [
                mngs.dsp.utils.design_filter(
                    seq_len, fs, low_hz=None, high_hz=cc, is_bandstop=False
                )
                for cc in cutoffs_hz
            ]
        )
        kernels = mngs.dsp.utils.ensure_even_len(kernels)
        return torch.tensor(kernels)


class HighPassFilter(BaseFilter1D):
    def __init__(self, cutoffs_hz, fs, x_shape):
        super().__init__()

        # Ensures bands shape
        assert cutoffs_hz.ndim == 1

        # Check bands definitions
        nyq = fs / 2.0
        for cc in cutoffs_hz:
            assert 0 < cc
            assert cc < nyq

        self.register_buffer(
            "kernels", self.init_kernels(x_shape[-1], fs, cutoffs_hz)
        )

    @staticmethod
    def init_kernels(seq_len, fs, cutoffs_hz):
        kernels = mngs.dsp.utils.zero_pad(
            [
                mngs.dsp.utils.design_filter(
                    seq_len, fs, low_hz=cc, high_hz=None, is_bandstop=False
                )
                for cc in cutoffs_hz
            ]
        )
        kernels = mngs.dsp.utils.ensure_even_len(kernels)
        return torch.tensor(kernels)


class GaussianFilter(BaseFilter1D):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = mngs.gen.to_even(sigma)
        self.register_buffer("kernels", self.init_kernels(sigma))

    @staticmethod
    def init_kernels(sigma):
        kernel_size = sigma * 6  # +/- 3SD
        kernel_range = torch.arange(0, kernel_size) - kernel_size // 2
        kernel = torch.exp(-0.5 * (kernel_range / sigma) ** 2)
        kernel /= kernel.sum()
        kernels = kernel.unsqueeze(0)  # n_filters = 1
        kernels = mngs.dsp.utils.ensure_even_len(kernels)
        return torch.tensor(kernels)


class DifferentiableBandPassFilter(BaseFilter1D):
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
        fp16=False,
    ):
        super().__init__(fp16=fp16)

        self.fp16 = fp16

        # Check bands definitions
        nyq = fs / 2.0
        assert pha_low_hz < pha_high_hz < nyq
        assert amp_low_hz < amp_high_hz < nyq

        # Prepare kernels
        self.init_kernels = mngs.nn.DifferentiableBandPassFilterInitializer
        kernels, self.pha_bands, self.amp_bands = self.init_kernels(
            sig_len,
            fs,
            pha_low_hz=pha_low_hz,
            pha_high_hz=pha_high_hz,
            pha_n_bands=pha_n_bands,
            amp_low_hz=amp_low_hz,
            amp_high_hz=amp_high_hz,
            amp_n_bands=amp_n_bands,
            cycle=cycle,
        )()

        if fp16:
            kernels = kernels.half()
        self.register_buffer(
            "kernels",
            kernels,
        )


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, fig_scale=5
    )

    xx, tt, fs = mngs.dsp.demo_sig(sig_type="chirp")
    xx = torch.tensor(xx).cuda()
    # bands = np.array([[2, 3], [3, 4]])
    # BandPassFilter(bands, fs, xx.shape)
    m = DifferentiableBandPassFilter(xx.shape[-1], fs).cuda()

    xf = m(xx)  # (8, 19, 80, 2048)

    m.pha_bands
    # Parameter containing:
    # tensor([ 2.0000,  2.6207,  3.2414,  3.8621,  4.4828,  5.1034,  5.7241,  6.3448,
    #          6.9655,  7.5862,  8.2069,  8.8276,  9.4483, 10.0690, 10.6897, 11.3103,
    #         11.9310, 12.5517, 13.1724, 13.7931, 14.4138, 15.0345, 15.6552, 16.2759,
    #         16.8966, 17.5172, 18.1379, 18.7586, 19.3793, 20.0000],
    #        requires_grad=True)
    m.amp_bands
    # Parameter containing:
    # tensor([ 80.0000,  81.6327,  83.2653,  84.8980,  86.5306,  88.1633,  89.7959,
    #          91.4286,  93.0612,  94.6939,  96.3265,  97.9592,  99.5918, 101.2245,
    #         102.8571, 104.4898, 106.1225, 107.7551, 109.3878, 111.0204, 112.6531,
    #         114.2857, 115.9184, 117.5510, 119.1837, 120.8163, 122.4490, 124.0816,
    #         125.7143, 127.3469, 128.9796, 130.6122, 132.2449, 133.8775, 135.5102,
    #         137.1429, 138.7755, 140.4082, 142.0408, 143.6735, 145.3061, 146.9388,
    #         148.5714, 150.2041, 151.8367, 153.4694, 155.1020, 156.7347, 158.3673,
    #         160.0000], requires_grad=True)

    xf.sum().backward()  # OK

    # PSD
    bands = torch.hstack([m.pha_bands, m.amp_bands])

    # Plots PSD
    # matplotlib.use("TkAgg")
    fig, axes = mngs.plt.subplots(nrows=1 + len(bands), ncols=2)

    psd, ff = mngs.dsp.psd(xx, fs)  # Orig
    axes[0, 0].plot(tt, xx[0, 0].detach().cpu().numpy(), label="orig")
    axes[0, 1].plot(
        ff.detach().cpu().numpy(),
        psd[0, 0].detach().cpu().numpy(),
        label="orig",
    )

    for i_filt in range(len(bands)):
        mid_hz = int(bands[i_filt].item())
        psd_f, ff_f = mngs.dsp.psd(xf[:, :, i_filt, :], fs)
        axes[i_filt + 1, 0].plot(
            tt,
            xf[0, 0, i_filt].detach().cpu().numpy(),
            label=f"filted at {mid_hz} Hz",
        )

        axes[i_filt + 1, 1].plot(
            ff_f.detach().cpu().numpy(),
            psd_f[0, 0].detach().cpu().numpy(),
            label=f"filted at {mid_hz} Hz",
        )
    for ax in axes.ravel():
        ax.legend(loc="upper left")

    mngs.io.save(fig, CONFIG["SDIR"] + "traces.png")
    # plt.show()

    # Close
    mngs.gen.close(CONFIG)

"""
/home/ywatanabe/proj/entrance/mngs/dsp/nn/_Filters.py
"""
