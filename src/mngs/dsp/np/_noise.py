#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-31 11:33:53 (ywatanabe)"

import matplotlib.pyplot as plt
import numpy as np


def unbias_1d(x):
    """
    Removes the mean from a 1D array to center it around zero.

    Parameters:
    - x (numpy.ndarray): Input 1D array.

    Returns:
    - numpy.ndarray: Unbiased 1D array.
    """
    assert x.ndim == 1
    return x - x.mean()


def normalize_1d(x, amp=1.0):
    """
    Normalizes a 1D array to have a specified amplitude.

    Parameters:
    - x (numpy.ndarray): Input 1D array.
    - amp (float): Target amplitude.

    Returns:
    - numpy.ndarray: Normalized 1D array.
    """
    assert x.ndim == 1
    return amp * x / max(abs(x.max()), abs(x.min()))


def white_1d(x, amp=1.0):
    """
    Adds white noise to the input signal.

    Parameters:
    - x (numpy.ndarray): Input signal.
    - amp (float): Amplitude of the white noise.

    Returns:
    - numpy.ndarray: Signal with added white noise.
    """
    assert x.ndim == 1
    return x + np.random.uniform(-amp, amp, x.shape)


def gauss_1d(x, amp=1.0):
    """
    Adds Gaussian noise to the input signal.

    Parameters:
    - x (numpy.ndarray): Input signal.
    - amp (float): Amplitude of the Gaussian noise.

    Returns:
    - numpy.ndarray: Signal with added Gaussian noise.
    """
    assert x.ndim == 1
    noise = np.random.rand(*x.shape)
    noise = unbias_1d(noise)
    return x + normalize_1d(noise, amp)


def brown_1d(x, amp=1.0):
    """
    Adds Brownian (Brown) noise to the input signal.

    Parameters:
    - x (numpy.ndarray): Input signal.
    - amp (float): Amplitude of the Brown noise.

    Returns:
    - numpy.ndarray: Signal with added Brown noise.
    """
    assert x.ndim == 1
    noise = np.cumsum(np.random.uniform(-1, 1, x.shape))
    return x + normalize_1d(unbias_1d(noise), amp)


def pink_1d(x, amp=1.0):
    """
    Adds Pink noise to the input signal.

    Parameters:
    - x (numpy.ndarray): Input signal.
    - amp (float): Amplitude of the Pink noise.

    Returns:
    - numpy.ndarray: Signal with added Pink noise.
    """
    assert x.ndim == 1
    cols = len(x)
    noise = np.random.randn(cols)
    noise = np.fft.rfft(noise)
    indices = np.arange(1, len(noise))
    noise[1:] /= np.sqrt(indices)
    noise = np.fft.irfft(noise, n=cols)
    return x + normalize_1d(unbias_1d(noise), amp)


if __name__ == "__main__":
    import mngs

    x = 0 * mngs.dsp.np.demo_sig_1d()
    amp = 1.0

    fig, axes = plt.subplots(nrows=5, sharex=True, sharey=True)
    axes[0].plot(x, label="orig")
    axes[1].plot(white_1d(x, amp=amp), label="white")
    axes[2].plot(gauss_1d(x, amp=amp), label="gauss")
    axes[3].plot(brown_1d(x, amp=amp), label="brown")
    axes[4].plot(pink_1d(x, amp=amp), label="pink")
    for ax in axes:
        ax.legend(loc="upper right")
    plt.show()
