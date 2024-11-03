#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 02:49:08 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/_converters.py

from typing import Callable, Optional

import pandas as pd
import xarray

"""
Functionality:
    - Provides core conversion utilities between different data types
    - Implements warning system for data type conversions
Input:
    - Various data types (NumPy, PyTorch, Pandas)
Output:
    - Converted data in target format
Prerequisites:
    - NumPy, PyTorch, Pandas packages
"""

import warnings
from functools import lru_cache
from typing import Any as _Any
from typing import Dict, Tuple, Union

import numpy as np
import torch


class ConversionWarning(UserWarning):
    pass


warnings.simplefilter("always", ConversionWarning)


@lru_cache(maxsize=None)
def _cached_warning(message: str) -> None:
    warnings.warn(message, category=ConversionWarning)


def _conversion_warning(old: _Any, new: torch.Tensor) -> None:
    message = (
        f"Converted from {type(old).__name__} to {type(new).__name__} ({new.device}). "
        f"Consider using {type(new).__name__} ({new.device}) as input for faster computation."
    )
    _cached_warning(message)


def _try_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.device.type == device:
        return tensor
    try:
        return tensor.to(device)
    except RuntimeError as error:
        if "cuda" in str(error).lower() and device == "cuda":
            warnings.warn("CUDA memory insufficient, falling back to CPU.", UserWarning)
            return tensor.cpu()
        raise error


def unsqueeze_if(
    arr_or_tensor: Union[np.ndarray, torch.Tensor],
    ndim: int = 2,
    axis: int = 0,
) -> torch.Tensor:
    if not isinstance(arr_or_tensor, (np.ndarray, torch.Tensor)):
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
    if arr_or_tensor.ndim != ndim:
        raise ValueError(f"Input must be a {ndim}-D array or tensor.")
    return torch.unsqueeze(arr_or_tensor, dim=axis)


def squeeze_if(
    arr_or_tensor: Union[np.ndarray, torch.Tensor],
    ndim: int = 3,
    axis: int = 0,
) -> torch.Tensor:
    if not isinstance(arr_or_tensor, (np.ndarray, torch.Tensor)):
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
    if arr_or_tensor.ndim != ndim:
        raise ValueError(f"Input must be a {ndim}-D array or tensor.")
    return torch.squeeze(arr_or_tensor, dim=axis)


def is_torch(*args: _Any, **kwargs: _Any) -> bool:
    return any(isinstance(arg, torch.Tensor) for arg in args) or any(
        isinstance(val, torch.Tensor) for val in kwargs.values()
    )


def is_cuda(*args: _Any, **kwargs: _Any) -> bool:
    return any((isinstance(arg, torch.Tensor) and arg.is_cuda) for arg in args) or any(
        (isinstance(val, torch.Tensor) and val.is_cuda) for val in kwargs.values()
    )


def _return_always(*args: _Any, **kwargs: _Any) -> Tuple[Tuple, Dict]:
    return args, kwargs


def _return_if(*args: _Any, **kwargs: _Any) -> Union[Tuple, Dict, None]:
    if args and kwargs:
        return args, kwargs
    elif args:
        return args
    elif kwargs:
        return kwargs
    else:
        return None


def to_torch(*args: _Any, return_fn: Callable = _return_if, **kwargs: _Any) -> _Any:
    def _to_torch(data: _Any, device: Optional[str] = kwargs.get("device")) -> _Any:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(data, (pd.Series, pd.DataFrame)):
            new_data = torch.tensor(data.to_numpy()).squeeze().float()
            new_data = _try_device(new_data, device)
            if device == "cuda":
                _conversion_warning(data, new_data)
            return new_data

        if isinstance(data, (np.ndarray, list)):
            new_data = torch.tensor(data).float()
            new_data = _try_device(new_data, device)
            if device == "cuda":
                _conversion_warning(data, new_data)
            return new_data

        if isinstance(data, xarray.core.dataarray.DataArray):
            new_data = torch.tensor(np.array(data)).float()
            new_data = _try_device(new_data, device)
            if device == "cuda":
                _conversion_warning(data, new_data)
            return new_data

        if isinstance(data, tuple):
            return [_to_torch(item) for item in data if item is not None]

        return data

    converted_args = [_to_torch(arg) for arg in args if arg is not None]
    converted_kwargs = {
        key: _to_torch(val) for key, val in kwargs.items() if val is not None
    }

    if "axis" in converted_kwargs:
        converted_kwargs["dim"] = converted_kwargs.pop("axis")

    return return_fn(*converted_args, **converted_kwargs)


def to_numpy(*args: _Any, return_fn: Callable = _return_if, **kwargs: _Any) -> _Any:
    def _to_numpy(data: _Any) -> _Any:
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data.to_numpy().squeeze()
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        if isinstance(data, list):
            return np.array(data)
        if isinstance(data, tuple):
            return [_to_numpy(item) for item in data if item is not None]
        return data

    converted_args = [_to_numpy(arg) for arg in args if arg is not None]
    converted_kwargs = {
        key: _to_numpy(val) for key, val in kwargs.items() if val is not None
    }

    if "dim" in converted_kwargs:
        converted_kwargs["axis"] = converted_kwargs.pop("dim")

    return return_fn(*converted_args, **converted_kwargs)


# EOF
