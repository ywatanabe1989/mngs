#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-03 00:15:24 (ywatanabe)"

import torch
import torch.nn as nn


class PSD(nn.Module):
    def __init__(self, sample_rate, dim=-1):
        super(PSD, self).__init__()
        self.sample_rate = sample_rate
        self.dim = dim

    def forward(self, signal):
        """
        Calculate the Power Spectral Density (PSD) of a batched multi-channel signal.

        Parameters:
        - signal (torch.Tensor): The time-domain signal tensor with shape (batch_size, n_chs, seq_len).

        Returns:
        - psd (torch.Tensor): The power spectral density of the signal with shape (batch_size, n_chs, seq_len//2).
        - freqs (torch.Tensor): The frequency bins with shape (seq_len//2,).
        """
        # Perform the FFT along the last dimension
        signal_fft = torch.fft.fft(signal, dim=self.dim)

        # Calculate the power spectrum
        power_spectrum = torch.abs(signal_fft) ** 2

        # Normalize the power spectrum
        power_spectrum = power_spectrum / signal.size(self.dim)

        # Get the corresponding frequency bins
        freqs = torch.fft.fftfreq(signal.size(self.dim), 1 / self.sample_rate)

        # Take the one-sided spectrum for real signal
        half = signal.size(self.dim) // 2
        psd = power_spectrum[..., :half]

        # Scale the power spectrum by the frequency resolution to get the PSD
        psd = psd * (1.0 / self.sample_rate)

        # Since freqs is symmetric, take the first half for the one-sided spectrum
        freqs = freqs[:half]
        self.freqs = freqs

        return psd, freqs
