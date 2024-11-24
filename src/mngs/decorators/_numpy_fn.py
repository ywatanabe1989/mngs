#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-25 00:28:14 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/_numpy_fn.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_numpy_fn.py"

"""
Functionality:
    - Provides decorator for automatic conversion between numpy arrays and other data types
    - Maintains type consistency for input/output operations
Input:
    - Python functions that expect numpy array inputs
    - Various data types (pandas DataFrames, torch tensors, lists, etc.)
Output:
    - Results in the same format as input (numpy array or torch tensor)
Prerequisites:
    - numpy package
    - Core converter utilities from _converters module
"""

from functools import wraps
from typing import Any as _Any
from typing import Callable

from ._converters import (
    _conversion_warning,
    _return_always,
    _return_if,
    _try_device,
    is_cuda,
    is_torch,
    to_numpy,
    to_torch,
)

def numpy_fn(func: Callable) -> Callable:
    """Decorates functions to handle numpy array conversions.

    Automatically converts input arguments to numpy arrays and handles output
    conversions based on input type (torch.Tensor or numpy.ndarray).

    Example
    -------
    >>> @numpy_fn
    ... def add_one(arr):
    ...     return arr + 1
    >>> tensor_data = torch.tensor([1, 2, 3])
    >>> result = add_one(tensor_data)
    >>> print(type(result), result)
    <class 'torch.Tensor'> tensor([2, 3, 4])

    Parameters
    ----------
    func : Callable
        Function that expects numpy array inputs

    Returns
    -------
    Callable
        Wrapped function that handles data type conversions
    """
    @wraps(func)
    def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
        is_torch_input = is_torch(*args, **kwargs)
        device = "cuda" if is_cuda(*args, **kwargs) else "cpu"

        converted_args, converted_kwargs = to_numpy(*args, return_fn=_return_always, **kwargs)
        results = func(*converted_args, **converted_kwargs)

        if is_torch_input:
            if isinstance(results, tuple):
                results = tuple(to_torch(r, return_fn=_return_if, device=device) for r in results)
            else:
                results = to_torch(results, return_fn=_return_if, device=device)
                if isinstance(results, tuple) and len(results) == 1:
                    results = results[0]
        return results

    return wrapper

def numpy_method(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args: _Any, **kwargs: _Any) -> _Any:
        is_torch_input = is_torch(*args, **kwargs)
        device = "cuda" if is_cuda(*args, **kwargs) else "cpu"
        converted_args, converted_kwargs = to_numpy(
            *args, return_fn=_return_always, **kwargs
        )
        results = func(self, *converted_args, **converted_kwargs)
        return (
            results
            if not is_torch_input
            else to_torch(results, return_fn=_return_if, device=device)[0][0]
        )
    return wrapper


# EOF
