#!/usr/bin/env python3

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from obspy.signal.tf_misfit import cwt


def compute_wavelet_transform(
    data, samp_rate, f_min=100, f_max=None, plot=False
):
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
        result = compute_wavelet_transform(wave, samp_rate, f_min=10, f_max=500, plot=True)
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


def wavelet(*args, **kwargs):
    """
    Deprecated: Use 'compute_wavelet_transform' instead.
    """
    warnings.warn(
        "'wavelet' is deprecated and will be removed in a future version. "
        "Use 'compute_wavelet_transform' instead.",
        DeprecationWarning,
    )
    return compute_wavelet_transform(*args, **kwargs)


# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from obspy.signal.tf_misfit import cwt


# def wavelet(wave, samp_rate, f_min=100, f_max=None, plot=False):
#     dt = 1.0 / samp_rate
#     npts = len(wave)
#     t = np.linspace(0, dt * npts, npts)
#     if f_min == None:
#         f_min = 0.1
#     if f_max == None:
#         f_max = int(samp_rate / 2)
#     scalogram = cwt(wave, dt, 8, f_min, f_max)

#     if plot:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         x, y = np.meshgrid(
#             t, np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0])
#         )

#         ax.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
#         ax.set_xlabel("Time [s]")
#         ax.set_ylabel("Frequency [Hz]")
#         ax.set_yscale("log")
#         ax.set_ylim(f_min, f_max)
#         plt.show()

#     Hz = pd.Series(np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))
#     df = pd.DataFrame(np.abs(scalogram))
#     df["Hz"] = Hz
#     df.set_index("Hz", inplace=True)

#     return df
