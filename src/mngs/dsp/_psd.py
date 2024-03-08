#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-21 08:58:35 (ywatanabe)"

import torch


def psd_torch(signal, samp_rate):
    """
    # Example usage:
    samp_rate = 480  # Sampling rate in Hz
    signal = torch.randn(480)  # Example signal with 480 samples
    freqs, psd = calculate_psd(signal, samp_rate)

    # Plot the PSD (if you have matplotlib installed)
    import matplotlib.pyplot as plt

    plt.plot(freqs.numpy(), psd.numpy())
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (V^2/Hz)')
    plt.title('Power Spectral Density')
    plt.show()
    """
    # Apply window function to the signal (e.g., Hanning window)
    window = torch.hann_window(signal.size(-1))
    signal = signal * window

    # Perform the FFT
    fft_output = torch.fft.fft(signal)

    # Compute the power spectrum (magnitude squared of the FFT output)
    power_spectrum = torch.abs(fft_output) ** 2

    # Normalize the power spectrum to get the PSD
    # Usually, we divide by the length of the signal and the sum of the window squared
    # to get the power in terms of physical units (e.g., V^2/Hz)
    psd = power_spectrum / (samp_rate * (window**2).sum())

    # Since the signal is real, we only need the positive half of the FFT output
    # The factor of 2 accounts for the energy in the negative frequencies that we're discarding
    psd = psd[: len(psd) // 2] * 2

    # Adjust the DC component (0 Hz) and Nyquist component (if applicable)
    psd[0] /= 2
    if len(psd) % 2 == 0:  # Even length, Nyquist freq component is included
        psd[-1] /= 2

    # Frequency axis
    freqs = torch.fft.fftfreq(signal.size(-1), 1 / samp_rate)[: len(psd)]

    return freqs, psd
