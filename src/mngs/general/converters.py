#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-20 17:22:43 (ywatanabe)"#!/usr/bin/env python3

import numpy as np

import pandas as pd
import torch


class KeepPD:
    def __init__(self):  # [REVISED]
        self.columns = None
        self.index = None
        self.i_called = 0

    def __call__(self, df):  # [REVISED]
        if self.i_called % 2 == 0:  # [REVISED]
            self.columns = df.columns.copy()
            self.index = df.index.copy()
        if self.i_called % 2 == 1:  # [REVISED]
            df.columns = self.columns
            df.index = self.index
        self.i_called += 1  # [REVISED]


def my2array(arr_or_tensor):
    """
    Convert a PyTorch tensor to a NumPy array. If the input is a NumPy array, pandas DataFrame, or Series, return it unchanged.

    Arguments:
        arr_or_tensor (np.ndarray, pd.DataFrame, pd.Series, or torch.Tensor): The input tensor, array, DataFrame, or Series to convert.

    Returns:
        np.ndarray: The input data as a NumPy array.
    """
    if isinstance(arr_or_tensor, torch.Tensor):
        # Convert PyTorch tensor to NumPy array
        return arr_or_tensor.numpy(), "torch"
    elif isinstance(arr_or_tensor, (np.ndarray, pd.DataFrame, pd.Series)):
        # Input is already a NumPy array, pandas DataFrame, or Series, return it as is
        return arr_or_tensor, "numpy"
    else:
        raise TypeError(
            "Input must be a PyTorch tensor, NumPy array, pandas DataFrame, or pandas Series."
        )


def my2tensor(arr_or_tensor):
    """
    Convert a NumPy array, pandas DataFrame, or Series to a PyTorch tensor. If the input is already a tensor, return it unchanged.

    Arguments:
        arr_or_tensor (np.ndarray, pd.DataFrame, pd.Series, or torch.Tensor): The input array, DataFrame, Series, or tensor to convert.

    Returns:
        torch.Tensor: The input data as a PyTorch tensor.
    """
    if isinstance(arr_or_tensor, (np.ndarray, pd.DataFrame, pd.Series)):
        # Convert NumPy array, pandas DataFrame, or Series to PyTorch tensor
        out_tensor = torch.tensor(
            arr_or_tensor.values
            if isinstance(arr_or_tensor, (pd.DataFrame, pd.Series))
            else arr_or_tensor
        )
        return out_tensor, "numpy"
    elif isinstance(arr_or_tensor, torch.Tensor):
        # Input is already a tensor, return it as is
        return arr_or_tensor, "torch"
    else:
        raise TypeError(
            "Input must be a NumPy array, pandas DataFrame, pandas Series, or a PyTorch tensor."
        )


def two2three_dim(arr_or_tensor, axis=0):
    """
    Add an extra dimension to a 2D array or tensor.

    Arguments:
        arr_or_tensor (np.ndarray or torch.Tensor): The input 2D array or tensor.
        axis (int): The axis along which to insert the new dimension.

    Returns:
        torch.Tensor: The input data as a 3D PyTorch tensor.
    """
    if isinstance(arr_or_tensor, (np.ndarray, torch.Tensor)):
        if arr_or_tensor.ndim != 2:
            raise ValueError("Input must be a 2D array or tensor.")
        # Add an extra dimension
        return torch.unsqueeze(arr_or_tensor, dim=axis)
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


def three2two_dim(arr_or_tensor, axis=0):
    """
    Remove a dimension from a 3D array or tensor.

    Arguments:
        arr_or_tensor (np.ndarray or torch.Tensor): The input 3D array or tensor.
        axis (int): The axis along which to remove the dimension.

    Returns:
        torch.Tensor: The input data as a 2D PyTorch tensor.
    """
    if isinstance(arr_or_tensor, (np.ndarray, torch.Tensor)):
        if arr_or_tensor.ndim != 3:
            raise ValueError("Input must be a 3D array or tensor.")
        # Remove the specified dimension
        return torch.squeeze(arr_or_tensor, dim=axis)
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
