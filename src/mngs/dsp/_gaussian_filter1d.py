#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-20 12:41:40 (ywatanabe)"

import numpy as np
import pandas as pd
import scipy.ndimage
import torch


def gaussian_filter1d(xx, radius):
    """
    Apply a one-dimensional Gaussian filter to an input array, tensor, or DataFrame.

    Arguments:
        xx (numpy.ndarray, torch.Tensor, or pandas.DataFrame): The input data to filter. It can be a 1D or 2D array/tensor/DataFrame.
        radius (int): The radius of the Gaussian kernel. The standard deviation of the Gaussian kernel is implicitly set to 1.

    Returns:
        numpy.ndarray, torch.Tensor, or pandas.DataFrame: The filtered data, with the same type as the input.

    Data Types:
        Input can be either numpy.ndarray, torch.Tensor, or pandas.DataFrame. Output will match the input data type.

    Data Shapes:
        - Input xx: If 1D, shape (n,), if 2D, shape (m, n)
        - Output: Same shape as input xx

    References:
        - SciPy documentation for gaussian_filter1d: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html
    """
    sigma = 1
    truncate = radius / sigma

    # Convert input to NumPy array if it is a pandas DataFrame or PyTorch tensor
    if isinstance(xx, pd.DataFrame):
        values = xx.values
        result = scipy.ndimage.gaussian_filter1d(
            values, sigma, truncate=truncate
        )
        return pd.DataFrame(result, index=xx.index, columns=xx.columns)
    elif isinstance(xx, torch.Tensor):
        values = xx.numpy()
        result = scipy.ndimage.gaussian_filter1d(
            values, sigma, truncate=truncate
        )
        return torch.from_numpy(result)
    else:
        # Assume input is a NumPy array
        return scipy.ndimage.gaussian_filter1d(xx, sigma, truncate=truncate)


# import scipy


# def gaussian_filter1d(xx, radius):
#     # radius = round(truncate * sigma)
#     sigma = 1
#     truncate = radius / sigma
#     return scipy.ndimage.gaussian_filter1d(xx, sigma, truncate=truncate)
