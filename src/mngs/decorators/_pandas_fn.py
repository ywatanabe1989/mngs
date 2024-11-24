#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-17 21:49:55 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/_pandas_fn.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py"

"""
Functionality:
    - Provides decorator for automatic conversion between pandas DataFrames and other data types
    - Maintains type consistency for input/output operations
Input:
    - Python functions that expect pandas DataFrame inputs
    - Various data types (numpy arrays, torch tensors, lists, etc.)
Output:
    - Results in the same format as input (pandas DataFrame, torch tensor, or numpy array)
Prerequisites:
    - pandas, numpy, and torch packages
    - Core converter utilities from _converters module
"""

from ._converters import (
    _conversion_warning,
    _return_if,
    is_torch,
    is_cuda,
    to_numpy,
    to_torch,
)
from functools import wraps

from typing import Any as _Any
from typing import Callable

import numpy as np
import pandas as pd
import torch


# def pandas_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         device = "cuda" if is_cuda(*args, **kwargs) else "cpu"

#         def to_pandas(data: _Any) -> pd.DataFrame:
#             if isinstance(data, pd.DataFrame):
#                 return data
#             elif isinstance(data, pd.Series):
#                 return pd.DataFrame(data)
#             elif isinstance(data, (np.ndarray, list)):
#                 return pd.DataFrame(data)
#             elif isinstance(data, torch.Tensor):
#                 return pd.DataFrame(data.detach().cpu().numpy())
#             else:
#                 return pd.DataFrame([data])

#         converted_args = [to_pandas(arg) for arg in args]
#         converted_kwargs = {key: to_pandas(val) for key, val in kwargs.items()}
#         results = func(*converted_args, **converted_kwargs)
#         if is_torch_input:
#             return to_torch(results, return_fn=_return_if, device=device)[0]
#         elif isinstance(results, (pd.DataFrame, pd.Series)):
#             return results
#         else:
#             return to_numpy(results, return_fn=_return_if)[0]

#     return wrapper


def pandas_fn(func: Callable) -> Callable:
    """Decorates functions to handle pandas DataFrame conversions.

    Automatically converts input arguments to pandas DataFrames and handles output
    conversions based on input type (torch.Tensor, numpy.ndarray, or pandas).

    Example
    -------
    >>> @pandas_fn
    ... def sum_values(data):
    ...     return data.sum()
    >>> tensor_data = torch.tensor([[1, 2], [3, 4]])
    >>> result = sum_values(tensor_data)
    >>> print(type(result), result)
    <class 'torch.Tensor'> tensor([10.])

    Parameters
    ----------
    func : Callable
        Function that expects pandas DataFrame inputs

    Returns
    -------
    Callable
        Wrapped function that handles data type conversions
    """
    @wraps(func)
    def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
        is_torch_input = is_torch(*args, **kwargs)
        device = "cuda" if is_cuda(*args, **kwargs) else "cpu"

        def to_pandas(data: _Any) -> pd.DataFrame:
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, pd.Series):
                return pd.DataFrame(data)
            elif isinstance(data, (np.ndarray, list)):
                return pd.DataFrame(data)
            elif isinstance(data, torch.Tensor):
                return pd.DataFrame(data.detach().cpu().numpy())
            else:
                return pd.DataFrame([data])

        converted_args = [to_pandas(arg) for arg in args]
        converted_kwargs = {key: to_pandas(val) for key, val in kwargs.items()}
        results = func(*converted_args, **converted_kwargs)

        if is_torch_input:
            if isinstance(results, tuple):
                results = tuple(
                    to_torch(r, return_fn=_return_if, device=device)
                    for r in results
                )
            else:
                results = to_torch(
                    results, return_fn=_return_if, device=device
                )
                if isinstance(results, tuple) and len(results) == 1:
                    results = results[0]
        elif isinstance(results, tuple):
            results = tuple(
                (
                    r
                    if isinstance(r, (pd.DataFrame, pd.Series))
                    else to_numpy(r, return_fn=_return_if)
                )
                for r in results
            )
        elif not isinstance(results, (pd.DataFrame, pd.Series)):
            results = to_numpy(results, return_fn=_return_if)
            if isinstance(results, tuple) and len(results) == 1:
                results = results[0]

        return results

    return wrapper


# EOF
