#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "ywatanabe (2024-11-03 00:11:11)"
# File: ./mngs_repo/src/mngs/decorators/_converters.py

"""
Functionality:
    - Provides utility functions and decorators for seamless data type conversions between NumPy, PyTorch, and Pandas
    - Implements batch processing functionality for large datasets
Input:
    - Various data types including NumPy arrays, PyTorch tensors, Pandas DataFrames, and Python lists
Output:
    - Converted data in the desired format (NumPy, PyTorch, or Pandas)
Prerequisites:
    - NumPy, PyTorch, Pandas, xarray, tqdm
"""

import warnings
from functools import lru_cache, wraps

import numpy as np
import pandas as pd
import torch
import xarray
from tqdm import tqdm as _tqdm

# from ..typing import Any, Callable, Dict, Optional, Tuple, Union
from typing import Any as _Any
from typing import Callable, Dict, Optional, Tuple, Union



class ConversionWarning(UserWarning):
    pass


@lru_cache(maxsize=None)
def _cached_warning(message: str) -> None:
    warnings.warn(message, category=ConversionWarning)


def _conversion_warning(old: _Any, new: torch.Tensor) -> None:
    message = (
        f"Converted from {type(old).__name__} to {type(new).__name__} ({new.device}). "
        f"Consider using {type(new).__name__} ({new.device}) as input for faster computation."
    )
    _cached_warning(message)


warnings.simplefilter("always", ConversionWarning)


def _try_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.device.type == device:
        return tensor
    try:
        return tensor.to(device)
    except RuntimeError as error:
        if "cuda" in str(error).lower() and device == "cuda":
            warnings.warn(
                "CUDA memory insufficient, falling back to CPU.", UserWarning
            )
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
    return any(
        (isinstance(arg, torch.Tensor) and arg.is_cuda) for arg in args
    ) or any(
        (isinstance(val, torch.Tensor) and val.is_cuda)
        for val in kwargs.values()
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


def to_torch(
    *args: _Any, return_fn: Callable = _return_if, **kwargs: _Any
) -> _Any:
    def _to_torch(
        data: _Any, device: Optional[str] = kwargs.get("device")
    ) -> _Any:
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


def to_numpy(
    *args: _Any, return_fn: Callable = _return_if, **kwargs: _Any
) -> _Any:
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


def torch_fn(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
        is_torch_input = is_torch(*args, **kwargs)
        converted_args, converted_kwargs = to_torch(
            *args, return_fn=_return_always, **kwargs
        )
        results = func(*converted_args, **converted_kwargs)
        return (
            to_numpy(results, return_fn=_return_if)[0]
            if not is_torch_input
            else results
        )

    return wrapper


def numpy_fn(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
        is_torch_input = is_torch(*args, **kwargs)
        device = "cuda" if is_cuda(*args, **kwargs) else "cpu"
        converted_args, converted_kwargs = to_numpy(
            *args, return_fn=_return_always, **kwargs
        )
        results = func(*converted_args, **converted_kwargs)
        return (
            results
            if not is_torch_input
            else to_torch(results, return_fn=_return_if, device=device)[0][0]
        )

    return wrapper


def batch_fn(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(x: _Any, *args: _Any, **kwargs: _Any) -> _Any:
        batch_size = int(kwargs.pop("batch_size", 4))
        if len(x) <= batch_size:
            return func(x, *args, **kwargs, batch_size=batch_size)
        n_batches = (len(x) + batch_size - 1) // batch_size
        results = []
        for i_batch in _tqdm(range(n_batches)):
            start = i_batch * batch_size
            end = min((i_batch + 1) * batch_size, len(x))
            batch_result = func(
                x[start:end], *args, **kwargs, batch_size=batch_size
            )
            if isinstance(batch_result, torch.Tensor):
                batch_result = batch_result.cpu()
            elif isinstance(batch_result, tuple):
                batch_result = tuple(
                    val.cpu() if isinstance(val, torch.Tensor) else val
                    for val in batch_result
                )
            results.append(batch_result)
        if isinstance(results[0], tuple):
            n_vars = len(results[0])
            combined_results = [
                torch.vstack([res[i_var] for res in results])
                for i_var in range(n_vars)
            ]
            return tuple(combined_results)
        elif isinstance(results[0], torch.Tensor):
            return torch.vstack(results)
        else:
            return sum(results, [])

    return wrapper


def pandas_fn(func: Callable) -> Callable:
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
            return to_torch(results, return_fn=_return_if, device=device)[0]
        elif isinstance(results, (pd.DataFrame, pd.Series)):
            return results
        else:
            return to_numpy(results, return_fn=_return_if)[0]

    return wrapper


if __name__ == "__main__":
    import scipy
    import torch.nn.functional as F

    @torch_fn
    def torch_softmax(*args: _Any, **kwargs: _Any) -> torch.Tensor:
        return F.softmax(*args, **kwargs)

    @numpy_fn
    def numpy_softmax(*args: _Any, **kwargs: _Any) -> np.ndarray:
        return scipy.special.softmax(*args, **kwargs)

    @pandas_fn
    def pandas_mean(*args: _Any, **kwargs: _Any) -> Union[pd.Series, float]:
        return pd.DataFrame(*args, **kwargs).mean()

    def custom_print(data: _Any) -> None:
        print(type(data), data)

    test_data = [1, 2, 3]
    test_list = test_data
    test_tensor = torch.tensor(test_data).float()
    test_tensor_cuda = torch.tensor(test_data).float().cuda()
    test_array = np.array(test_data)
    test_df = pd.DataFrame({"col1": test_data})

    print("Testing torch_fn:")
    custom_print(torch_softmax(test_list, dim=-1))
    custom_print(torch_softmax(test_array, dim=-1))
    custom_print(torch_softmax(test_df, dim=-1))
    custom_print(torch_softmax(test_tensor, dim=-1))
    custom_print(torch_softmax(test_tensor_cuda, dim=-1))

    print("\nTesting numpy_fn:")
    custom_print(numpy_softmax(test_list, axis=-1))
    custom_print(numpy_softmax(test_array, axis=-1))
    custom_print(numpy_softmax(test_df, axis=-1))
    custom_print(numpy_softmax(test_tensor, axis=-1))
    custom_print(numpy_softmax(test_tensor_cuda, axis=-1))

    print("\nTesting pandas_fn:")
    custom_print(pandas_mean(test_list))
    custom_print(pandas_mean(test_array))
    custom_print(pandas_mean(test_df))
    custom_print(pandas_mean(test_tensor))
    custom_print(pandas_mean(test_tensor_cuda))


# EOF
