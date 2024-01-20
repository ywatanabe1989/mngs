#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-20 12:50:41 (ywatanabe)"

import warnings

import numpy as np
import pandas as pd
import torch


def subtract_random_reference(data, random_state=42):
    """
    Subtract a random reference from each channel/row of the input data.

    This function shuffles the channels/rows of the input data and subtracts
    the shuffled data from the original data. This can be used in signal processing
    to remove common noise shared across channels.

    Arguments:
        data (numpy.ndarray, torch.Tensor, or pandas.DataFrame): The input 2D data
            where each row represents a channel and each column represents a time point.
        random_state (int, optional): A seed for the random number generator to ensure
            reproducibility. Defaults to 42.

    Returns:
        The data with a random reference subtracted from each channel/row. The returned
        data will have the same type and shape as the input data.

    Data Types and Shapes:
        - If the input is a numpy.ndarray, the output will also be a numpy.ndarray with the same shape.
        - If the input is a torch.Tensor, the output will also be a torch.Tensor with the same shape.
        - If the input is a pandas.DataFrame, the output will also be a pandas.DataFrame with the same shape.

    Examples:
        # Using numpy arrays
        numpy_array = np.random.rand(16, 1000)
        subtracted_array = subtract_random_reference(numpy_array)

        # Using PyTorch tensors
        torch_tensor = torch.randn(16, 1000)
        subtracted_tensor = subtract_random_reference(torch_tensor)

        # Using pandas DataFrames
        pandas_df = pd.DataFrame(np.random.rand(16, 1000))
        subtracted_df = subtract_random_reference(pandas_df)
    """
    if isinstance(data, np.ndarray):
        rs = np.random.RandomState(random_state)
        ref_data = data[rs.permutation(data.shape[0])]
        return data - ref_data

    elif isinstance(data, torch.Tensor):
        torch.manual_seed(random_state)
        ref_data = data[torch.randperm(data.size(0))]
        return data - ref_data

    elif isinstance(data, pd.DataFrame):
        rs = np.random.RandomState(random_state)
        ref_data = data.iloc[rs.permutation(data.index)]
        return data - ref_data

    else:
        raise TypeError(
            "Input must be a numpy.ndarray, torch.Tensor, or pandas.DataFrame"
        )


def take_random_references(*args, **kwargs):
    """
    Deprecated: Use 'subtract_random_reference' instead.
    """
    warnings.warn(
        "'take_random_references' is deprecated and will be removed in a future version. "
        "Use 'subtract_random_reference' instead.",
        DeprecationWarning,
    )
    return subtract_random_reference(*args, **kwargs)


# def subtract_random_reference(data, random_state=42):
#     """
#     Subtract a random reference from each channel/row of the input data.

#     This function shuffles the channels/rows of the input data and subtracts
#     the shuffled data from the original data. This can be used in signal processing
#     to remove common noise shared across channels.

#     Arguments:
#         data (numpy.ndarray or torch.Tensor or pandas.DataFrame): The input 2D data
#             where each row represents a channel and each column represents a time point.
#         random_state (int, optional): A seed for the random number generator to ensure
#             reproducibility. Defaults to 42.

#     Returns:
#         numpy.ndarray or torch.Tensor or pandas.DataFrame: The data with a random reference
#             subtracted from each channel/row. The returned data will have the same type
#             and shape as the input data.

#     Data Types and Shapes:
#         - If the input is a numpy.ndarray, the output will also be a numpy.ndarray with the same shape.
#         - If the input is a torch.Tensor, the output will also be a torch.Tensor with the same shape.
#         - If the input is a pandas.DataFrame, the output will also be a pandas.DataFrame with the same shape.

#     Examples:
#         # Using numpy arrays
#         numpy_array = np.random.rand(16, 1000)
#         subtracted_array = subtract_random_reference(numpy_array)

#         # Using PyTorch tensors
#         torch_tensor = torch.randn(16, 1000)
#         subtracted_tensor = subtract_random_reference(torch_tensor)

#         # Using pandas DataFrames
#         pandas_df = pd.DataFrame(np.random.rand(16, 1000))
#         subtracted_df = subtract_random_reference(pandas_df)
#     """
#     if isinstance(data, np.ndarray):
#         rs = np.random.RandomState(random_state)
#         ref_data = data[rs.permutation(data.shape[0])]
#         return data - ref_data

#     elif isinstance(data, torch.Tensor):
#         rs = torch.manual_seed(random_state)
#         ref_data = data[torch.randperm(data.size(0))]
#         return data - ref_data

#     elif isinstance(data, pd.DataFrame):
#         rs = np.random.RandomState(random_state)
#         ref_data = data.iloc[rs.permutation(data.index)]
#         return data - ref_data

#     else:
#         raise TypeError("Input must be a numpy.ndarray, torch.Tensor, or pandas.DataFrame")

# # import numpy as np


# # def take_random_references(sig_2D, random_state=42):
# #     n_chs = len(sig_2D)
# #     rs = np.random.RandomState(random_state)
# #     ref_sig_2D = sig_2D[rs.permutation(np.arange(n_chs))]
# #     return sig_2D - ref_sig_2D
