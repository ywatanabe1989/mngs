#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-20 13:58:43 (ywatanabe)"#!/usr/bin/env python3

import numpy as np
import torch


def demo_sig_np(
    T_sec=50.0, fs=250, freqs_hz=[30, 60, 100, 200, 1000], n_chs=19
):
    """
    Prepare a demo dataset composed of multiple channels of sinusoidal waves.

    Parameters:
        T_sec (float): The duration of the signal in seconds. Defaults to 50.0 seconds.
        fs (int): The sampling frequency of the signal in samples per second. Defaults to 250 Hz.
        freqs_hz (list of float): A list of frequencies in Hz for the sinusoidal components. Defaults to [30, 60, 100, 200, 1000].
        n_chs (int): The number of channels to generate. Defaults to 19.

    Returns:
        numpy.ndarray: A 2D array representing the generated multi-channel signal, with shape (n_chs, T_sec * fs).

    Data Types:
        Input:
            T_sec: float
            fs: int
            freqs_hz: list of float
            n_chs: int
        Output:
            numpy.ndarray (typically float64)

    Data Shapes:
        Input:
            freqs_hz: (n_frequencies,)
        Output:
            numpy.ndarray: (n_chs, n_samples)

    References:
        - Sinusoidal wave: https://en.wikipedia.org/wiki/Sine_wave
        - NumPy array: https://numpy.org/doc/stable/reference/generated/numpy.array.html

    Examples:
        # Prepare a demo dataset with a 10-second duration, sampled at 500 Hz, with frequencies 50, 100, and 150 Hz, and 16 channels
        demo_data = prepair_demo_data(T_sec=10.0, fs=500, freqs_hz=[50, 100, 150], n_chs=16)
    """
    data = np.array(
        [
            summarize_sinusoidal_waves(T_sec=T_sec, fs=fs, freqs_hz=freqs_hz)
            for _ in range(n_chs)
        ]
    )
    return data


def demo_sig_torch(
    batch_size=64, n_chs=19, samp_rate=1000, len_sec=10, freqs_hz=[2, 5, 10]
):
    time = torch.arange(0, len_sec, 1 / samp_rate)
    sig = torch.vstack(
        [torch.sin(f * 2 * torch.pi * time) for f in freqs_hz]
    ).sum(dim=0)
    sig = sig.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_chs, 1)
    return sig


def summarize_sinusoidal_waves(
    T_sec=50.0, fs=250, freqs_hz=[30, 60, 100, 200, 1000]
):
    """
    Generate a signal composed of a sum of sinusoidal waves with random amplitudes and phases.

    Parameters:
        T_sec (float): The duration of the signal in seconds. Defaults to 50.0 seconds.
        fs (int): The sampling frequency of the signal in samples per second. Defaults to 250 Hz.
        freqs_hz (list of float): A list of frequencies in Hz for the sinusoidal components. Defaults to [30, 60, 100, 200, 1000].

    Returns:
        numpy.ndarray: The generated signal, which is a 1D array of length `T_sec * fs`.

    Data Types:
        Input:
            T_sec: float
            fs: int
            freqs_hz: list of float
        Output:
            numpy.ndarray (typically float64)

    Data Shapes:
        Input:
            freqs_hz: (n_frequencies,)
        Output:
            numpy.ndarray: (n_samples,)

    References:
        - Sinusoidal wave: https://en.wikipedia.org/wiki/Sine_wave
        - NumPy linspace: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        - NumPy random.rand: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html

    Examples:
        # Generate a 10-second signal sampled at 500 Hz with frequencies 50, 100, and 150 Hz
        signal = summarize_sinusoidal_waves(T_sec=10.0, fs=500, freqs_hz=[50, 100, 150])
    """

    n = int(T_sec * fs)
    t = np.linspace(0, T_sec, n, endpoint=False)
    summed = np.array(
        [
            np.random.rand()
            * np.sin((f_hz * t + np.random.rand()) * (2 * np.pi))
            for f_hz in freqs_hz
        ]
    ).sum(axis=0)
    return summed
