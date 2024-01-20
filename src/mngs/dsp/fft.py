#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
from scipy import fftpack

# def psd(signals_2d, samp_rate, normalize=True):
#     return calc_fft_amps(signals_2d, samp_rate, normalize=normalize)


def fft_powers(signals_2d, samp_rate, normalize=True):
    """
    Calculate the power spectrum of the FFT (Fast Fourier Transform) for each signal in a 2D array or tensor.

    Arguments:
        signals_2d (numpy.ndarray or torch.Tensor): A 2D array or tensor containing multiple signals,
                                                     where each row represents a signal.
        samp_rate (float): The sampling rate of the signals.
        normalize (bool, optional): If True, normalize the FFT powers by the sum of powers for each signal.
                                    Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing the FFT powers for each signal. The columns represent the
                          frequencies, and the rows represent the individual signals.

    Data Types:
        Input can be either numpy.ndarray or torch.Tensor. Output is always a pandas.DataFrame.

    Data Shapes:
        - signals_2d: (n_signals, signal_length)
        - Output DataFrame: (n_signals, n_frequencies)

    Example:
        sig_len = 1024
        n_sigs = 32
        signals_2d = np.random.rand(n_sigs, sig_len)
        samp_rate = 256
        fft_powers_df = calc_fft_powers(signals_2d, samp_rate)

    References:
        - NumPy documentation for FFT: https://numpy.org/doc/stable/reference/routines.fft.html
        - PyTorch documentation for FFT: https://pytorch.org/docs/stable/fft.html
        - pandas.DataFrame documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    """

    return calc_fft_amps_2d(signals_2d, samp_rate, normalize=normalize)


# def fft_amps(signals_2d, samp_rate, normalize=True):
#     """
#     Example:
#         sig_len = 1024
#         n_sigs = 32
#         signals_2d = np.random.rand(n_sigs, sig_len)
#         samp_rate = 256
#         fft_df = calc_fft_amps(signals_2d, samp_rate)
#     """
#     fft_amps = np.abs(fftpack.fft(signals_2d))
#     fft_freqs = np.fft.fftfreq(signals_2d.shape[-1], d=1.0 / samp_rate)
#     mask = fft_freqs >= 0
#     fft_amps, fft_freqs = fft_amps[:, mask], np.round(fft_freqs[mask], 1)

#     if normalize == True:
#         fft_amps = fft_amps / np.sum(fft_amps, axis=1, keepdims=True)
#         # fft_outs[0] / np.sum(np.array(fft_outs[0]), axis=1, keepdims=True)

#     fft_df = pd.DataFrame(data=fft_amps, columns=fft_freqs.astype(str))
#     return fft_df


def fft_amps(signals_2d, samp_rate, normalize=True):
    """
    Calculate the amplitude of the FFT (Fast Fourier Transform) for each signal in a 2D array or tensor.

    Arguments:
        signals_2d (numpy.ndarray or torch.Tensor): A 2D array or tensor containing multiple signals,
                                                     where each row represents a signal.
        samp_rate (float): The sampling rate of the signals.
        normalize (bool, optional): If True, normalize the FFT amplitudes by the sum of amplitudes for each signal.
                                    Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing the FFT amplitudes for each signal. The columns represent the
                          frequencies, and the rows represent the individual signals.

    Data Types:
        Input can be either numpy.ndarray or torch.Tensor. Output is always a pandas.DataFrame.

    Data Shapes:
        - signals_2d: (n_signals, signal_length)
        - Output DataFrame: (n_signals, n_frequencies)

    Example:
        sig_len = 1024
        n_sigs = 32
        signals_2d = np.random.rand(n_sigs, sig_len)
        samp_rate = 256
        fft_df = calc_fft_amps(signals_2d, samp_rate)

    """
    if isinstance(signals_2d, torch.Tensor):
        signals_2d = (
            signals_2d.detach().cpu().numpy()
        )  # Convert to NumPy array if input is a PyTorch tensor

    fft_amps = np.abs(fftpack.fft(signals_2d))
    fft_freqs = np.fft.fftfreq(signals_2d.shape[-1], d=1.0 / samp_rate)
    mask = fft_freqs >= 0
    fft_amps, fft_freqs = fft_amps[:, mask], np.round(fft_freqs[mask], 1)

    if normalize:
        fft_amps = fft_amps / np.sum(fft_amps, axis=1, keepdims=True)

    fft_df = pd.DataFrame(data=fft_amps, columns=fft_freqs.astype(str))
    return fft_df
