#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-20 12:47:15 (ywatanabe)"

import numpy as np
import pandas as pd
import torch


def normalize_time(x, time_dim=-1):
    """
    Normalize the input data along the specified time dimension.

    The function subtracts the mean and divides by the standard deviation along
    the specified axis for numpy.ndarray and torch.Tensor, or along the columns
    for pandas.DataFrame.

    Args:
        x (numpy.ndarray or torch.Tensor or pandas.DataFrame): The input data to normalize.
        time_dim (int, optional): The dimension along which to normalize the data.
                                  Defaults to -1 (the last dimension).

    Returns:
        numpy.ndarray or torch.Tensor or pandas.DataFrame: The normalized data with the same type as the input.

    Examples:
        numpy_array = np.random.rand(16, 160, 1000)
        normalized_array = normalize_time(numpy_array, time_dim=2)

        torch_tensor = torch.randn(16, 160, 1000)
        normalized_tensor = normalize_time(torch_tensor, time_dim=2)

        pandas_dataframe = pd.DataFrame(np.random.rand(160, 1000))
        normalized_dataframe = normalize_time(pandas_dataframe)  # time_dim is ignored for DataFrame
    """
    if isinstance(x, torch.Tensor):
        mean = x.mean(dim=time_dim, keepdim=True)
        std = x.std(dim=time_dim, keepdim=True)
        return (x - mean) / std

    elif isinstance(x, np.ndarray):
        mean = x.mean(axis=time_dim, keepdims=True)
        std = x.std(axis=time_dim, keepdims=True)
        return (x - mean) / std

    elif isinstance(x, pd.DataFrame):
        if time_dim != -1 and time_dim != 1:
            raise ValueError(
                "For pandas.DataFrame, time_dim must be -1 or 1 (columns)."
            )
        mean = x.mean(axis=1).values.reshape(-1, 1)
        std = x.std(axis=1).values.reshape(-1, 1)
        return (x - mean) / std

    else:
        raise TypeError(
            "Input must be a numpy.ndarray, torch.Tensor, or pandas.DataFrame"
        )


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
