#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-03 01:06:30 (ywatanabe)"

import math
import warnings

import mngs
import numpy as np
import pywt
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
        self.kernel = None

    @property
    def kernel_size(
        self,
    ):
        return mngs.gen.to_even(self.kernel.shape[-1])

    @property
    def radius(
        self,
    ):
        # return mngs.gen.to_even(self.kernel_size // 2)
        return self.kernel_size // 2

    def forward(self, x):
        """Apply the filter to input signal x with shape: (batch_size, n_chs, seq_len)"""

        x = mngs.dsp.ensure_3d(x)
        seq_len = x.shape[-1]

        # Ensure the kernel is initialized
        if self.kernel is None:
            self.init_kernel()
            if self.kernel is None:
                raise ValueError("Filter kernel has not been initialized.")

        # Edge handling and convolution
        extension_length = self.radius
        first_segment = x[:, :, :extension_length].flip(dims=[-1])
        last_segment = x[:, :, -extension_length:].flip(dims=[-1])
        extended_x = torch.cat([first_segment, x, last_segment], dim=-1)

        channels = extended_x.size(1)

        kernel = (
            self.kernel.expand(channels, 1, -1)
            .to(extended_x.device)
            .to(extended_x.dtype)
        )

        filtered_extended_x = F.conv1d(
            extended_x, kernel, padding=0, groups=channels
        )[..., :seq_len]

        assert x.shape == filtered_extended_x.shape

        return filtered_extended_x


class GaussianFilter(BaseFilter1D):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = mngs.gen.to_even(sigma)
        self.init_kernel()

    def init_kernel(self):
        # Create a Gaussian kernel
        kernel_size = self.sigma * 6  # +/- 3SD
        kernel_range = torch.arange(kernel_size) - kernel_size // 2
        kernel = torch.exp(-0.5 * (kernel_range / self.sigma) ** 2)
        kernel /= kernel.sum()  # Normalize the kernel
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)


class BandPassFilter(BaseFilter1D):
    def __init__(self, low_hz, high_hz, fs, kernel_size=None):
        super().__init__()

        assert 0 < low_hz
        assert low_hz < high_hz
        assert high_hz <= fs / 2

        kernel_size = (
            mngs.gen.to_even(int(1 / low_hz * fs * 3))
            if kernel_size is None
            else mngs.gen.to_even(kernel_size)
        )

        self.low_hz = low_hz
        self.high_hz = high_hz
        self.fs = fs
        self.init_kernel(kernel_size)

    def init_kernel(self, kernel_size):
        freqs = torch.fft.fftfreq(kernel_size, d=1 / self.fs)
        kernel = torch.zeros(kernel_size)
        kernel[(freqs >= self.low_hz) & (freqs <= self.high_hz)] = 1
        kernel = torch.fft.ifft(kernel).real
        kernel /= kernel.sum()
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)


class BandStopFilter(BaseFilter1D):
    def __init__(self, low_hz, high_hz, fs, kernel_size=None):
        super().__init__()
        kernel_size = (
            mngs.gen.to_even(int(1 / low_hz * fs * 3))
            if kernel_size is None
            else mngs.gen.to_even(kernel_size)
        )
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.fs = fs
        self.init_kernel(kernel_size)

    def init_kernel(self, kernel_size):
        freqs = torch.fft.fftfreq(kernel_size, d=1 / self.fs)
        kernel = torch.ones(kernel_size)
        kernel[(freqs >= self.low_hz) & (freqs <= self.high_hz)] = 0
        kernel = torch.fft.ifft(
            kernel
        ).real  # Inverse FFT to get the time-domain kernel
        kernel /= kernel.sum()  # Normalize the kernel
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)
