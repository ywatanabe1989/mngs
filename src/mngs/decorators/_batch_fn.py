#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-24 19:28:37 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/_batch_fn.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_batch_fn.py"

"""
Functionality:
    Provides decorators for batch processing of data with PyTorch
Input:
    Functions that process data arrays
Output:
    Batch-processed results combined appropriately
Prerequisites:
    torch, tqdm
"""

from typing import List, Tuple, Union
import numpy as _np
from functools import wraps
from typing import Any as _Any
from typing import Callable

import torch
from tqdm import tqdm as _tqdm
from functools import wraps
from typing import Any as _Any, Callable, List, Tuple, Union

import torch
from tqdm import tqdm as _tqdm



# ## Working on arr2ftrs.py but not for describe.py
# def _combine_results(
#     results: List[Union[torch.Tensor, _np.ndarray, Tuple, List]]
# ) -> Union[torch.Tensor, _np.ndarray, Tuple, List]:
#     """Combines batch results based on their type."""
#     if isinstance(results[0], torch.Tensor):
#         return torch.vstack(results)
#     elif isinstance(results[0], _np.ndarray):
#         return _np.vstack(results)
#     elif isinstance(results[0], tuple):
#         n_vars = len(results[0])
#         return tuple(
#             torch.vstack([r[i] for r in results]) for i in range(n_vars)
#         )
#     return _np.vstack(results)


def _combine_results(
    results: List[Union[torch.Tensor, _np.ndarray, Tuple, List]]
) -> Union[torch.Tensor, _np.ndarray, Tuple, List]:
    """Combines batch results based on their type."""
    if not results:
        return None

    if isinstance(results[0], torch.Tensor):
        try:
            return torch.cat(results, dim=0)
        except:
            return torch.stack(results, dim=0)
    elif isinstance(results[0], _np.ndarray):
        try:
            return _np.concatenate(results, axis=0)
        except:
            return _np.stack(results, axis=0)
    elif isinstance(results[0], tuple):
        return tuple(
            _combine_results([r[i] for r in results])
            for i in range(len(results[0]))
        )
    elif isinstance(results[0], list):
        return sum(results, [])  # Flatten list of lists
    return results


def _process_batch_data(
    func: Callable,
    data: Union[List, torch.Tensor, _np.ndarray],
    batch_size: Union[int, List[int]],
    desc: str,
    *args: _Any,
    **kwargs: _Any,
) -> Union[List, torch.Tensor, _np.ndarray, Tuple[torch.Tensor, ...]]:
    """Processes data in batches along first N dimensions based on batch_size."""
    if not hasattr(data, 'shape'):
        raise TypeError("Input must have shape attribute")

    batch_sizes = [batch_size] if isinstance(batch_size, int) else batch_size

    n_dims_to_batch = len(batch_sizes)

    data_sizes = data.shape[:n_dims_to_batch]

    if all(d <= b for d, b in zip(data_sizes, batch_sizes)):
        return func(data, *args, **kwargs, batch_size=batch_sizes)

    n_batches = [((s + b - 1) // b) for s, b in zip(data_sizes, batch_sizes)]
    results = []

    for idx in _tqdm(range(int(_np.prod(n_batches))), desc=desc):
        multi_idx = _np.unravel_index(idx, n_batches)

        slice_obj = [slice(None)] * data.ndim
        for dim in range(n_dims_to_batch):
            start = multi_idx[dim] * batch_sizes[dim]
            end = min((multi_idx[dim] + 1) * batch_sizes[dim], data_sizes[dim])
            slice_obj[dim] = slice(start, end)

        batch_data = data[tuple(slice_obj)]
        batch_result = func(
            batch_data, *args, **kwargs, batch_size=batch_sizes
        )
        results.append(batch_result)

    return _combine_results(results)


def batch_fn(func: Callable) -> Callable:
    """Decorator for processing large data arrays in batches along multiple dimensions.

    Parameters
    ----------
    func : Callable
        Function to be decorated that accepts data array and returns processed results

    Returns
    -------
    Callable
        Wrapped function that handles batch processing

    Example
    -------
    >>> @batch_fn
    ... def process_data(data):
    ...     return data.mean(dim=-1)
    >>> data = torch.ones(1000, 500, 32)
    >>> # Process first two dimensions in batches
    >>> result = process_data(data, batch_size=[100, 50])
    >>> print(result.shape)
    torch.Size([1000, 500])

    Notes
    -----
    - If single batch_size is provided, processes only the first dimension
    - If list of batch_sizes is provided, processes corresponding dimensions in order
    """

    @wraps(func)
    def wrapper(data: Union[List, torch.Tensor], *args: _Any, **kwargs: _Any):
        batch_size = kwargs.pop("batch_size", -1)
        if isinstance(batch_size, (int, float)):
            batch_size = int(batch_size)
            # print(
            #     f"\nBatch size: {batch_size} (See `mngs.decorators.batch_fn`)"
            # )
        else:
            batch_size = [int(b) for b in batch_size]
            # print(
            #     f"\nBatch sizes: {batch_size} (See `mngs.decorators.batch_fn`)"
            # )

        if batch_size == -1:
            batch_size = len(data)
        desc = f"Processing data with shape {data.shape} in batches of {batch_size}"
        return _process_batch_data(
            func, data, batch_size, desc, *args, **kwargs
        )

    return wrapper


def batch_method(func: Callable) -> Callable:
    """Decorator for processing large data arrays in batches within class methods.

    Parameters
    ----------
    func : Callable
        Method to be decorated

    Returns
    -------
    Callable
        Wrapped method that handles batch processing

    Example
    -------
    >>> class DataProcessor:
    ...     @batch_method
    ...     def multiply(self, data, factor=2):
    ...         return data * factor
    >>> processor = DataProcessor()
    >>> data = torch.ones(10, 5)
    >>> result = processor.multiply(data, batch_size=2)
    >>> print(result)
    tensor([[2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.],
            [2., 2., 2., 2., 2.]])
    """

    @wraps(func)
    def wrapper(
        self, data: Union[List, torch.Tensor], *args: _Any, **kwargs: _Any
    ):
        batch_size = kwargs.pop("batch_size", -1)
        # Handle both single and multiple batch sizes
        if isinstance(batch_size, (int, float)):
            batch_size = int(batch_size)
            # print(
            #     f"\nBatch size: {batch_size} (See `mngs.decorators.batch_method`)"
            # )
        else:
            batch_size = [int(b) for b in batch_size]
            # print(
            #     f"\nBatch sizes: {batch_size} (See `mngs.decorators.batch_method`)"
            # )

        if batch_size == -1:
            batch_size = len(data)
        desc = f"Processing {len(data)} items in batches of {batch_size}"
        return _process_batch_data(
            lambda *a, **k: func(self, *a, **k),
            data,
            batch_size,
            desc,
            *args,
            **kwargs,
        )

    return wrapper


if __name__ == "__main__":
    run_main()

    import numpy as np
    from mngs.decorators import batch_fn

    @batch_fn
    def sum_data(data, batch_size=None):
        return data.sum(axis=1)

    data = np.ones((10, 5, 3))
    result = sum_data(data, batch_size=2)
    print(result)
    print(result.shape)


# EOF
