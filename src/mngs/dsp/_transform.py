#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-04 13:34:33 (ywatanabe)"#!/usr/bin/env python3


import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import torch
from mngs.general import torch_fn
from obspy.signal.tf_misfit import cwt


def to_sktime_df(arr):
    """
    Convert a 3D numpy array into a DataFrame suitable for sktime.

    Parameters:
    arr (numpy.ndarray): A 3D numpy array with shape (n_samples, n_channels, seq_len)

    Returns:
    pandas.DataFrame: A DataFrame in sktime format
    """
    if len(arr.shape) != 3:
        raise ValueError("Input data must be a 3D array")

    n_samples, seq_len, n_channels = arr.shape

    # Initialize an empty DataFrame for sktime format
    sktime_df = pd.DataFrame(index=range(n_samples), columns=["dim_0"])

    # Iterate over each sample
    for i in range(n_samples):
        # Combine all channels into a single cell
        combined_series = pd.Series(
            {
                f"channel_{j}": pd.Series(arr[i, :, j])
                for j in range(n_channels)
            }
        )
        sktime_df.iloc[i, 0] = combined_series

    return sktime_df


@torch_fn
def to_segments(x, window_size, overlap_factor=1, dim=-1):
    stride = window_size // overlap_factor
    num_windows = (x.size(dim) - window_size) // stride + 1
    windows = x.unfold(dim, window_size, stride)
    return windows


if __name__ == "__main__":
    x, t, f = mngs.dsp.demo_sig()

    y = to_segments(x, 256)

    x = 100 * np.random.rand(16, 160, 1000)
    print(_normalize_time(x))

    x = torch.randn(16, 160, 1000)
    print(_normalize_time(x))

    x = torch.randn(16, 160, 1000).cuda()
    print(_normalize_time(x))
