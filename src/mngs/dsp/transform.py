#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-24 05:56:27 (ywatanabe)"#!/usr/bin/env python3


import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import torch
from obspy.signal.tf_misfit import cwt


def wavelet(data, samp_rate, f_min=100, f_max=None, plot=False):
    """
    Compute the continuous wavelet transform (CWT) of a signal and optionally plot the scalogram.

    Arguments:
        data (numpy.ndarray | torch.Tensor | pandas.DataFrame): The input signal waveform.
        samp_rate (float): The sampling rate of the signal in Hz.
        f_min (float, optional): The minimum frequency of interest in Hz. Defaults to 100.
        f_max (float, optional): The maximum frequency of interest in Hz. Defaults to half the sampling rate.
        plot (bool, optional): If True, plot the scalogram. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the absolute values of the CWT coefficients,
                          indexed by frequency in Hz.

    Data Types and Shapes:
        - The input `data` should be a 1D numpy.ndarray, torch.Tensor, or pandas.Series.
        - The output is a pandas.DataFrame with frequencies as the index and time as the columns.

    References:
        - ObsPy's cwt function: https://docs.obspy.org/packages/autogen/obspy.signal.tf_misfit.cwt.html

    Examples:
        # Generate a sample signal
        samp_rate = 1000  # Sampling rate in Hz
        t = np.linspace(0, 1, samp_rate)
        wave = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 80 * t)

        # Compute the wavelet transform
        result = wavelet(wave, samp_rate, f_min=10, f_max=500, plot=True)
        print(result)
    """
    # Convert input data to a NumPy array if it is not already one
    if isinstance(data, torch.Tensor):
        wave = data.numpy()
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        wave = data.values.squeeze()
    else:
        wave = data

    # Ensure the input wave is a 1D array
    if wave.ndim != 1:
        raise ValueError("Input data must be a 1D array or Series.")

    dt = 1.0 / samp_rate
    npts = len(wave)
    t = np.linspace(0, dt * npts, npts)
    if f_min is None:
        f_min = 0.1
    if f_max is None:
        f_max = int(samp_rate / 2)
    scalogram = cwt(wave, dt, 8, f_min, f_max)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x, y = np.meshgrid(
            t,
            np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]),
        )

        ax.pcolormesh(x, y, np.abs(scalogram), shading="auto", cmap="viridis")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_yscale("log")
        ax.set_ylim(f_min, f_max)
        plt.show()

    Hz = pd.Series(
        np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0])
    )
    df = pd.DataFrame(np.abs(scalogram), columns=t)
    df["Hz"] = Hz
    df.set_index("Hz", inplace=True)

    return df


# def wavelet(*args, **kwargs):
#     """
#     Deprecated: Use 'compute_wavelet_transform' instead.
#     """
#     warnings.warn(
#         "'wavelet' is deprecated and will be removed in a future version. "
#         "Use 'compute_wavelet_transform' instead.",
#         DeprecationWarning,
#     )
#     return compute_wavelet_transform(*args, **kwargs)


def to_z(x, i_dim=-1):
    """
    Normalize the input data along the specified time dimension.

    The function subtracts the mean and divides by the standard deviation along
    the specified axis for numpy.ndarray and torch.Tensor, or along the columns
    for pandas.DataFrame.

    Args:
        x (numpy.ndarray or torch.Tensor or pandas.DataFrame): The input data to normalize.
        i_dim (int, optional): The dimension along which to normalize the data.
                                  Defaults to -1 (the last dimension).

    Returns:
        numpy.ndarray or torch.Tensor or pandas.DataFrame: The normalized data with the same type as the input.

    Examples:
        numpy_array = np.random.rand(16, 160, 1000)
        normalized_array = normalize_time(numpy_array, i_dim=2)

        torch_tensor = torch.randn(16, 160, 1000)
        normalized_tensor = normalize_time(torch_tensor, i_dim=2)

        pandas_dataframe = pd.DataFrame(np.random.rand(160, 1000))
        normalized_dataframe = normalize_time(pandas_dataframe)  # i_dim is ignored for DataFrame
    """
    if isinstance(x, torch.Tensor):
        mean = x.mean(dim=i_dim, keepdim=True)
        std = x.std(dim=i_dim, keepdim=True)
        return (x - mean) / std

    elif isinstance(x, np.ndarray):
        mean = x.mean(axis=i_dim, keepdims=True)
        std = x.std(axis=i_dim, keepdims=True)
        return (x - mean) / std

    elif isinstance(x, pd.DataFrame):
        if i_dim != -1 and i_dim != 1:
            raise ValueError(
                "For pandas.DataFrame, i_dim must be -1 or 1 (columns)."
            )
        mean = x.mean(axis=1).values.reshape(-1, 1)
        std = x.std(axis=1).values.reshape(-1, 1)
        return (x - mean) / std

    else:
        raise TypeError(
            "Input must be a numpy.ndarray, torch.Tensor, or pandas.DataFrame"
        )


def wavelet_np(wave, samp_rate, f_min=100, f_max=None, plot=False, title=None):
    """
    Perform a continuous wavelet transform on a signal and optionally plot the scalogram.

    Parameters:
        wave (numpy.ndarray): The input signal. Shape should be (n_samples,).
        samp_rate (float): The sampling rate of the signal in Hz.
        f_min (float): The minimum frequency for the wavelet transform. Defaults to 100 Hz.
        f_max (float, optional): The maximum frequency for the wavelet transform. Defaults to half the sampling rate.
        plot (bool): If True, plot the scalogram. Defaults to False.
        title (str, optional): The title for the plot if plot is True.

    Returns:
        pandas.DataFrame: A DataFrame containing the wavelet transform scalogram with frequencies as the index.

    Data Types:
        Input:
            wave: numpy.ndarray (typically float32 or float64)
            samp_rate: float
            f_min: float
            f_max: float or None
            plot: bool
            title: str or None
        Output:
            pandas.DataFrame

    Data Shapes:
        Input:
            wave: (n_samples,)
        Output:
            DataFrame: (n_frequencies, n_samples)

    References:
        - Continuous wavelet transform: https://en.wikipedia.org/wiki/Continuous_wavelet_transform
        - scipy.signal.cwt: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html
        - obspy.imaging.cm.obspy_sequential: https://docs.obspy.org/packages/autogen/obspy.imaging.cm.obspy_sequential.html
        - pandas.DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
        - matplotlib.pyplot: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html

    Raises:
        ValueError: If the input signal is not a 1D numpy.ndarray.
    """

    dt = 1.0 / samp_rate
    npts = len(wave)
    t = np.linspace(0, dt * npts, npts)
    if f_min is None:
        f_min = 0.1
    if f_max is None:
        f_max = int(samp_rate / 2)
        scalogram = cwt(wave, dt, 8, f_min, f_max)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x, y = np.meshgrid(
            t,
            np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]),
        )

        ax.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylim(f_min, f_max)
        fig.show()

    Hz = pd.Series(
        np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0])
    )
    df = pd.DataFrame(np.abs(scalogram))
    df["Hz"] = Hz
    df.set_index("Hz", inplace=True)

    return df


def psd(signals_2d, samp_rate, normalize=True):
    """
    Calculate the Power Spectral Density (PSD) of each signal in a 2D array, tensor, or DataFrame.

    Arguments:
        signals_2d (numpy.ndarray, torch.Tensor, or pandas.DataFrame): A 2D collection containing multiple signals,
                                                                       where each row represents a signal.
        samp_rate (float): The sampling rate of the signals.
        normalize (bool, optional): If True, normalize the PSD by the sum of powers for each signal.
                                    Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing the PSD for each signal. The columns represent the
                          frequencies, and the rows represent the individual signals.

    Data Types:
        Input can be either numpy.ndarray, torch.Tensor, or pandas.DataFrame. Output is always a pandas.DataFrame.

    Data Shapes:
        - signals_2d: (n_signals, signal_length)
        - Output DataFrame: (n_signals, n_frequencies)

    References:
        - NumPy documentation for FFT: https://numpy.org/doc/stable/reference/routines.fft.html
        - PyTorch documentation for FFT: https://pytorch.org/docs/stable/fft.html
        - pandas.DataFrame documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    """
    # Convert input to NumPy array if it is a pandas DataFrame or PyTorch tensor
    if isinstance(signals_2d, pd.DataFrame):
        signals_2d = signals_2d.values
    elif isinstance(signals_2d, torch.Tensor):
        signals_2d = signals_2d.numpy()

    # Calculate the FFT for each signal
    fft_results = np.fft.rfft(signals_2d, axis=1)

    # Calculate the PSD by taking the absolute square of the FFT results
    psd = np.abs(fft_results) ** 2

    # Normalize the PSD if requested
    if normalize:
        psd /= np.sum(psd, axis=1, keepdims=True)

    # Calculate the frequency bins
    freqs = np.fft.rfftfreq(signals_2d.shape[1], d=1 / samp_rate)

    # Create a DataFrame for the PSD with frequency labels
    psd_df = pd.DataFrame(data=psd, columns=freqs)

    return psd_df


def arr2skdf(data):
    """
    Convert a 3D numpy array into a DataFrame suitable for sktime.

    Parameters:
    data (numpy.ndarray): A 3D numpy array with shape (n_samples, n_channels, seq_len)

    Returns:
    pandas.DataFrame: A DataFrame in sktime format
    """
    if len(data.shape) != 3:
        raise ValueError("Input data must be a 3D array")

    n_samples, seq_len, n_channels = data.shape

    # Initialize an empty DataFrame for sktime format
    sktime_df = pd.DataFrame(index=range(n_samples), columns=["dim_0"])

    # Iterate over each sample
    for i in range(n_samples):
        # Combine all channels into a single cell
        combined_series = pd.Series(
            {
                f"channel_{j}": pd.Series(data[i, :, j])
                for j in range(n_channels)
            }
        )
        sktime_df.iloc[i, 0] = combined_series

    return sktime_df


def crop(data, window_size_pts, overlap_factor=1):
    assert data.ndim == 2

    cropped = skimage.util.view_as_windows(
        data,
        (len(data), window_size_pts),
        int(window_size_pts / overlap_factor),
    )

    if cropped.ndim != 3:
        cropped = cropped.squeeze(0)
    assert cropped.ndim == 3

    return cropped


# def normalize_time(x):
#     if type(x) == torch.Tensor:
#         return (x - x.mean(dim=-1, keepdims=True)) \
#             / x.std(dim=-1, keepdims=True)
#     if type(x) == np.ndarray:
#         return (x - x.mean(axis=-1, keepdims=True)) \
#             / x.std(axis=-1, keepdims=True)

if __name__ == "__main__":
    x = 100 * np.random.rand(16, 160, 1000)
    print(_normalize_time(x))

    x = torch.randn(16, 160, 1000)
    print(_normalize_time(x))

    x = torch.randn(16, 160, 1000).cuda()
    print(_normalize_time(x))
