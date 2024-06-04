#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-03 08:33:33 (ywatanabe)"#!/usr/bin/env python3

import warnings
from functools import wraps

import numpy as np
import pandas as pd
import torch
import xarray
from tqdm import tqdm


def try_device(x, device):
    if not isinstance(x, torch.Tensor):
        return x

    # When no conversion is necessary
    orig_device = x.device
    if orig_device == device:
        return x

    try:
        x_new = x.to(device)
        return x_new  # When successful

    except RuntimeError as e:
        if "cuda" in str(e).lower() and device == "cuda":
            warnings.warn(
                "CUDA memory insufficient, falling back to CPU.",
                UserWarning,
            )
            x_new = x.cpu()
            return x_new
        else:
            raise e


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
    def _conversion_warning(x, x_new):
        warnings.warn(
            f"Converted from {type(x).__name__} to {type(x_new).__name__} ({x_new.device}). "
            f"You might want to consider using {type(x_new).__name__} ({x_new.device}) as input for faster computation.",
            category=UserWarning,
        )

    def _to_torch(x, device=kwargs.get("device")):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Pandas
        if isinstance(x, (pd.Series, pd.DataFrame)):
            x_new = torch.tensor(x.to_numpy()).squeeze().float()
            x_new = try_device(x_new, device)
            if device == "cuda":
                _conversion_warning(x, x_new)
            return x_new

        # Numpy
        elif isinstance(x, (np.ndarray, list)):
            x_new = torch.tensor(x).float()
            x_new = try_device(x_new, device)
            if device == "cuda":
                _conversion_warning(x, x_new)
            return x_new

        # xarray
        elif isinstance(x, xarray.core.dataarray.DataArray):
            x_new = torch.tensor(np.array(x)).float()
            x_new = try_device(x_new, device)
            if device == "cuda":
                _conversion_warning(x, x_new)
            return x_new

        # Tuple
        elif isinstance(x, tuple):
            return [_to_torch(_x) for _x in x if _x is not None]

        else:
            x_new = x
            return x_new

    # def _to_torch(x, device=kwargs.get("device", "cuda")):
    #     # pandas
    #     if isinstance(x, (pd.Series, pd.DataFrame)):
    #         x = x.to_numpy()

    #     # numpy
    #     if isinstance(x, (np.ndarray, list)):
    #         x = torch.tensor(x)
    #         x_new = try_device(x, device)
    #         return x_new

    #     # Else
    #     else:
    #         # Directly return non-convertible types without warning
    #         return x

    c_args = [_to_torch(arg) for arg in args if arg is not None]
    c_kwargs = {k: _to_torch(v) for k, v in kwargs.items() if v is not None}

    # Handle renaming 'axis' to 'dim' for PyTorch compatibility
    if "axis" in c_kwargs:
        c_kwargs["dim"] = c_kwargs.pop("axis")

    return return_fn(*c_args, **c_kwargs)


# def to_torch(*args, return_fn=return_if, **kwargs):
#     def _to_torch(x, device=kwargs.get("device")):
#         if device is None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"

#         if isinstance(x, (pd.Series, pd.DataFrame)):
#             x_new = torch.tensor(x.to_numpy()).squeeze().float().to(device)
#             warnings.warn(
#                 f"Converted from  {type(x).__name__} to {type(x_new).__name__} ({x_new.device}) If you prioritize calculation speed, please consider to input with torch.tensor instead of {type(x).__name__}",
#                 category=UserWarning,
#             )
#             return x_new

#         elif isinstance(x, (np.ndarray, list)):
#             x_new = torch.tensor(x).float().to(device)
#             warnings.warn(
#                 f"Converted from  {type(x).__name__} to {type(x_new).__name__} ({x_new.device})",
#                 category=UserWarning,
#             )
#             return x_new

#         else:
#             return x

#     c_args = [_to_torch(arg) for arg in args if arg is not None]
#     c_kwargs = {k: _to_torch(v) for k, v in kwargs.items() if v is not None}

#     if "axis" in c_kwargs:
#         c_kwargs["dim"] = c_kwargs.pop("axis")

#     return return_fn(*c_args, **c_kwargs)


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
        else:
            return to_torch(results, return_fn=return_if, device=device)[0][0]

    return wrapper


def batch_fn(func):
    """
    Decorator to apply batch processing to a function. This is useful when the input tensor is too large to process at once due to memory constraints.

    Args:
    - func (Callable): A function that processes a tensor.

    Returns:
    - Callable: A wrapped function that processes the tensor in batches.
    """

    @wraps(func)
    def wrapper(x, *args, **kwargs):
        # Get batch_size from kwargs with a default value of 4
        batch_size = int(kwargs.pop("batch_size", 4))

        if len(x) <= batch_size:
            return func(x, *args, **kwargs, batch_size=batch_size)

        n_batches = (len(x) + batch_size - 1) // batch_size
        results = []

        for i_batch in tqdm(range(n_batches)):
            start = i_batch * batch_size
            end = min((i_batch + 1) * batch_size, len(x))
            batch_result = func(
                x[start:end], *args, **kwargs, batch_size=batch_size
            )

            if isinstance(batch_result, torch.Tensor):
                batch_result = batch_result.cpu()
            elif isinstance(batch_result, tuple):
                batch_result = tuple(
                    vv.cpu() if isinstance(vv, torch.Tensor) else vv
                    for vv in batch_result
                )

            results.append(batch_result)

        # Check if the function returns a tuple of results or a single result
        if isinstance(results[0], tuple):
            # Handle multiple outputs
            n_vars = len(results[0])
            combined_results = [
                torch.vstack([res[i_var] for res in results])
                for i_var in range(n_vars)
            ]
            return tuple(combined_results)
        else:
            # Handle single output
            if isinstance(results[0], torch.Tensor):
                return torch.vstack(results)
            else:
                # If the single output is not a tensor, concatenate or combine them as needed
                return sum(results, [])

    return wrapper


# def batch_fn(batch_size=4):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(x, *args, **kwargs):
#             if len(x) <= batch_size:
#                 return func(x, *args, **kwargs)

#             n_batches = (len(x) + batch_size - 1) // batch_size
#             results = []

#             for i_batch in tqdm(range(n_batches)):
#                 start = i_batch * batch_size
#                 end = min((i_batch + 1) * batch_size, len(x))
#                 batch_result = func(x[start:end], *args, **kwargs)

#                 if isinstance(batch_result, torch.Tensor):
#                     batch_result = batch_result.cpu()
#                 elif isinstance(batch_result, tuple):
#                     batch_result = tuple(
#                         vv.cpu() if isinstance(vv, torch.Tensor) else vv
#                         for vv in batch_result
#                     )

#                 results.append(batch_result)

#             # Check if the function returns a tuple of results or a single result
#             if isinstance(results[0], tuple):
#                 # Handle multiple outputs
#                 n_vars = len(results[0])
#                 combined_results = [
#                     torch.vstack([res[i_var] for res in results])
#                     for i_var in range(n_vars)
#                 ]
#                 return tuple(combined_results)
#             else:
#                 # Handle single output
#                 if isinstance(results[0], torch.Tensor):
#                     return torch.vstack(results)
#                 else:
#                     # If the single output is not a tensor, concatenate or combine them as needed
#                     return sum(results, [])

#         return wrapper

#     return decorator


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
