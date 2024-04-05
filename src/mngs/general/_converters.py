#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-04 13:22:40 (ywatanabe)"#!/usr/bin/env python3

import warnings
from functools import wraps

import numpy as np
import pandas as pd
import torch


def unsqueeze_if(arr_or_tensor, ndim=2, axis=0):
    if isinstance(arr_or_tensor, (np.ndarray, torch.Tensor)):
        if arr_or_tensor.ndim != ndim:
            raise ValueError(f"Input must be a {ndim}-D array or tensor.")
        return torch.unsqueeze(arr_or_tensor, dim=axis)
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


def squeeze_if(arr_or_tensor, ndim=3, axis=0):
    if isinstance(arr_or_tensor, (np.ndarray, torch.Tensor)):
        if arr_or_tensor.ndim != ndim:
            raise ValueError(f"Input must be a {ndim}-D array or tensor.")
        return torch.squeeze(arr_or_tensor, dim=axis)
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


def is_torch(*args, **kwargs):
    return any(isinstance(arg, torch.Tensor) for arg in args) or any(
        isinstance(v, torch.Tensor) for v in kwargs.values()
    )


def is_cuda(*args, **kwargs):
    return any(
        [(isinstance(arg, torch.Tensor) and arg.is_cuda) for arg in args]
    ) or any(
        [(isinstance(v, torch.Tensor) and v.is_cuda) for v in kwargs.values()]
    )


def return_always(*args, **kwargs):
    return args, kwargs


def return_if(*args, **kwargs):
    if (len(args) > 0) and (len(kwargs) > 0):
        return args, kwargs
    elif (len(args) > 0) and (len(kwargs) == 0):
        return args
    elif (len(args) == 0) and (len(kwargs) > 0):
        return kwargs
    elif (len(args) == 0) and (len(kwargs) == 0):
        return None


def to_torch(*args, return_fn=return_if, **kwargs):
    def _to_torch(x, device=kwargs.get("device")):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(x, (pd.Series, pd.DataFrame)):
            x_new = torch.tensor(x.to_numpy()).squeeze().float().to(device)
            warnings.warn(
                f"Converted from  {type(x)} to {type(x_new)} ({x_new.device})",
                category=UserWarning,
            )
            return x_new

        elif isinstance(x, (np.ndarray, list)):
            x_new = torch.tensor(x).float().to(device)
            warnings.warn(
                f"Converted from  {type(x)} to {type(x_new)} ({x_new.device})",
                category=UserWarning,
            )
            return x_new

        else:
            return x

    c_args = [_to_torch(arg) for arg in args if arg is not None]
    c_kwargs = {k: _to_torch(v) for k, v in kwargs.items() if v is not None}

    if "axis" in c_kwargs:
        c_kwargs["dim"] = c_kwargs.pop("axis")

    return return_fn(*c_args, **c_kwargs)


def to_numpy(*args, return_fn=return_if, **kwargs):
    def _to_numpy(x):
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return x.to_numpy().squeeze()
        elif isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, list):
            return np.array(x)
        elif isinstance(x, tuple):
            return [_to_numpy(_x) for _x in x if _x is not None]
        else:
            return x

    c_args = [_to_numpy(arg) for arg in args if arg is not None]
    c_kwargs = {k: _to_numpy(v) for k, v in kwargs.items() if v is not None}

    if "dim" in c_kwargs:
        c_kwargs["axis"] = c_kwargs.pop("dim")

    return return_fn(*c_args, **c_kwargs)


def torch_fn(func):
    """
    Decorator to ensure torch calculation

    Example:
        import torch.nn.functional as F

        @torch_fn
        def torch_softmax(*args, **kwargs):
            return F.softmax(*args, **kwargs)

        def custom_print(x):
            print(type(x), x)

        x = [1, 2, 3]
        x_list = x
        x_tensor = torch.tensor(x).float()
        x_tensor_cuda = torch.tensor(x).float().cuda()
        x_array = np.array(x)
        x_df = pd.DataFrame({"col1": x})

        custom_print(torch_softmax(x_list, dim=-1))
        custom_print(torch_softmax(x_array, dim=-1))
        custom_print(torch_softmax(x_df, dim=-1))
        custom_print(torch_softmax(x_tensor, dim=-1))
        custom_print(torch_softmax(x_tensor_cuda, dim=-1))

        # <class 'numpy.ndarray'> [0.09003057 0.24472848 0.66524094]
        # <class 'numpy.ndarray'> [0.09003057 0.24472848 0.66524094]
        # <class 'numpy.ndarray'> [0.09003057 0.24472848 0.66524094]
        # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652])
        # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652], device='cuda:0')
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Buffers the input types
        is_torch_input = is_torch(*args, **kwargs)
        # is_cuda_input = is_cuda(*args, **kwargs)

        # Runs the func
        c_args, c_kwargs = to_torch(*args, return_fn=return_always, **kwargs)

        results = func(*c_args, **c_kwargs)

        # Reverts to the original data type
        if not is_torch_input:
            return to_numpy(results, return_fn=return_if)[0]
        elif is_torch_input:
            return results

    return wrapper


def numpy_fn(func):
    """
    Decorator to ensure numpy calculation

    Example:
        import scipy

        @numpy_fn
        def numpy_softmax(*args, **kwargs):
            return scipy.special.softmax(*args, **kwargs)

        def custom_print(x):
            print(type(x), x)

        x = [1, 2, 3]
        x_list = x
        x_tensor = torch.tensor(x).float()
        x_tensor_cuda = torch.tensor(x).float().cuda()
        x_array = np.array(x)
        x_df = pd.DataFrame({"col1": x})

        custom_print(numpy_softmax(x_list, axis=-1))
        custom_print(numpy_softmax(x_array, axis=-1))
        custom_print(numpy_softmax(x_df, axis=-1))
        custom_print(numpy_softmax(x_tensor, axis=-1))
        custom_print(numpy_softmax(x_tensor_cuda, axis=-1))

        # <class 'numpy.ndarray'> [0.09003057 0.24472847 0.66524096]
        # <class 'numpy.ndarray'> [0.09003057 0.24472847 0.66524096]
        # <class 'numpy.ndarray'> [0.09003057 0.24472847 0.66524096]
        # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652])
        # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652], device='cuda:0')
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Buffers the input types
        is_torch_input = is_torch(*args, **kwargs)
        is_cuda_input = is_cuda(*args, **kwargs)
        device = "cuda" if is_cuda_input else "cpu"

        # Runs the func
        c_args, c_kwargs = to_numpy(*args, return_fn=return_always, **kwargs)
        results = func(*c_args, **c_kwargs)

        # Reverts to the original data type
        if not is_torch_input:
            return results
        elif is_torch_input:
            if not is_cuda_input:
                return to_torch(results, return_fn=return_if, device=device)[0]
            elif is_cuda_input:
                return to_torch(results, return_fn=return_if, device=device)[0]

    return wrapper


if __name__ == "__main__":
    import scipy
    import torch.nn.functional as F

    @torch_fn
    def torch_softmax(*args, **kwargs):
        return F.softmax(*args, **kwargs)

    @numpy_fn
    def numpy_softmax(*args, **kwargs):
        return scipy.special.softmax(*args, **kwargs)

    def custom_print(x):
        print(type(x), x)

    # Test the decorator with different input types
    x = [1, 2, 3]
    x_list = x
    x_tensor = torch.tensor(x).float()
    x_tensor_cuda = torch.tensor(x).float().cuda()
    x_array = np.array(x)
    x_df = pd.DataFrame({"col1": x})

    custom_print(torch_softmax(x_list, dim=-1))
    custom_print(torch_softmax(x_array, dim=-1))
    custom_print(torch_softmax(x_df, dim=-1))
    custom_print(torch_softmax(x_tensor, dim=-1))
    custom_print(torch_softmax(x_tensor_cuda, dim=-1))
    # <class 'numpy.ndarray'> [0.09003057 0.24472848 0.66524094]
    # <class 'numpy.ndarray'> [0.09003057 0.24472848 0.66524094]
    # <class 'numpy.ndarray'> [0.09003057 0.24472848 0.66524094]
    # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652])
    # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652], device='cuda:0')

    custom_print(numpy_softmax(x_list, axis=-1))
    custom_print(numpy_softmax(x_array, axis=-1))
    custom_print(numpy_softmax(x_df, axis=-1))
    custom_print(numpy_softmax(x_tensor, axis=-1))
    custom_print(numpy_softmax(x_tensor_cuda, axis=-1))
    # <class 'numpy.ndarray'> [0.09003057 0.24472847 0.66524096]
    # <class 'numpy.ndarray'> [0.09003057 0.24472847 0.66524096]
    # <class 'numpy.ndarray'> [0.09003057 0.24472847 0.66524096]
    # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652])
    # <class 'torch.Tensor'> tensor([0.0900, 0.2447, 0.6652], device='cuda:0')
