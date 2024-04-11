#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-11 00:11:16 (ywatanabe)"

import math
import warnings
from abc import ABC, abstractmethod

import mngs
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from mngs.general import torch_fn


class BaseFilter1D(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

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
        return torch.tensor(kernels)


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


if __name__ == "__main__":
    xx, tt, fs = mngs.dsp.demo_sig()
    bands = np.array([[2, 3], [3, 4]])
    fs = 32
    BandPassFilter(bands, fs, xx.shape)
