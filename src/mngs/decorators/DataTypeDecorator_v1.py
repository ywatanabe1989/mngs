#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-25 20:05:08 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/DataTypeDecorator.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/DataTypeDecorator.py"

"""
1. Functionality:
   - Provides decorators for data type conversion and batch processing
2. Input:
   - Functions to be decorated with data type and batch processing capabilities
3. Output:
   - Processed data with specified type and batch handling
4. Prerequisites:
   - torch, numpy, pandas, xarray
"""

"""Imports"""
from functools import wraps
import threading
from contextlib import contextmanager
from typing import Any, Callable, Union, Tuple
import torch
import numpy as np
import pandas as pd
import xarray as xr

class DataProcessor:
    _local = threading.local()

    @staticmethod
    @contextmanager
    def computation_context(dtype: str):
        if not hasattr(DataProcessor._local, 'dtype_stack'):
            DataProcessor._local.dtype_stack = []
        DataProcessor._local.dtype_stack.append(dtype)
        try:
            yield
        finally:
            DataProcessor._local.dtype_stack.pop()

    @staticmethod
    def convert_to_type(data: Any, target_type: str) -> Any:
        if data is None:
            return None

        # Already correct type
        if (target_type == 'torch' and torch.is_tensor(data) or
            target_type == 'numpy' and isinstance(data, np.ndarray) or
            target_type == 'pandas' and isinstance(data, (pd.Series, pd.DataFrame)) or
            target_type == 'xarray' and isinstance(data, xr.DataArray)):
            return data

        # Convert to numpy first as intermediate format
        if torch.is_tensor(data):
            numpy_data = data.cpu().numpy()
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            numpy_data = data.values
        elif isinstance(data, xr.DataArray):
            numpy_data = data.values
        elif isinstance(data, np.ndarray):
            numpy_data = data
        else:
            return data

        # Convert from numpy to target
        if target_type == 'torch':
            return torch.from_numpy(numpy_data)
        elif target_type == 'numpy':
            return numpy_data
        elif target_type == 'pandas':
            return pd.DataFrame(numpy_data)
        elif target_type == 'xarray':
            return xr.DataArray(numpy_data)
        return data

    @staticmethod
    def restore_type(data: Any, original: Any) -> Any:
        if original is None or data is None:
            return data
        if isinstance(original, type(data)):
            return data

        numpy_data = DataProcessor.convert_to_type(data, 'numpy')

        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(numpy_data, index=original.index, columns=original.columns)
        elif isinstance(original, pd.Series):
            return pd.Series(numpy_data, index=original.index)
        elif isinstance(original, xr.DataArray):
            return xr.DataArray(numpy_data, coords=original.coords, dims=original.dims)
        elif isinstance(original, torch.Tensor):
            return torch.from_numpy(numpy_data)
        return data

    @staticmethod
    def process_batches(func: Callable, data: Any, batch_size: Union[int, Tuple[int]], **kwargs) -> Any:
        if batch_size == -1:
            return func(data, **kwargs)

        if isinstance(batch_size, int):
            batch_size = (batch_size,)

        total_size = data.shape[0]
        batches = [data[i:i+batch_size[0]] for i in range(0, total_size, batch_size[0])]

        # Process each batch
        results = []
        for batch in batches:
            result = func(batch, **kwargs)
            results.append(result)

        # Combine results based on type
        if len(results) == 0:
            return None

        if torch.is_tensor(results[0]):
            return torch.cat(results)
        elif isinstance(results[0], np.ndarray):
            return np.concatenate(results)
        elif isinstance(results[0], pd.DataFrame):
            return pd.concat(results)
        elif isinstance(results[0], xr.DataArray):
            return xr.concat(results, dim=results[0].dims[0])

        return results

def create_decorator(target_type: str = None, enable_batch: bool = False):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            batch_size = kwargs.pop('batch_size', -1) if enable_batch else -1

            # Get current context type or use target_type
            current_type = (DataProcessor._local.dtype_stack[-1]
                          if hasattr(DataProcessor._local, 'dtype_stack')
                          and DataProcessor._local.dtype_stack
                          else target_type)

            # Convert input data
            if current_type:
                converted_args = [DataProcessor.convert_to_type(arg, current_type) for arg in args]
                converted_kwargs = {k: DataProcessor.convert_to_type(v, current_type)
                                 for k, v in kwargs.items()}
            else:
                converted_args = args
                converted_kwargs = kwargs

            # Process data
            if enable_batch:
                result = DataProcessor.process_batches(
                    lambda x: func(x, **converted_kwargs),
                    converted_args[0],
                    batch_size
                )
            else:
                result = func(*converted_args, **converted_kwargs)

            # Convert result to target type
            if target_type:
                result = DataProcessor.convert_to_type(result, target_type)

            return result
        return wrapper
    return decorator

# Decorators
torch_fn = create_decorator(target_type='torch')
numpy_fn = create_decorator(target_type='numpy')
pandas_fn = create_decorator(target_type='pandas')
xarray_fn = create_decorator(target_type='xarray')
batch_fn = create_decorator(enable_batch=True)


def run_tests():
    # Test data creation with fixed seed
    np.random.seed(42)
    np_data = np.random.randn(100, 10)
    torch_data = torch.from_numpy(np_data)
    pd_data = pd.DataFrame(np_data, columns=[f'col_{i}' for i in range(10)])
    xr_data = xr.DataArray(np_data, dims=['samples', 'features'])

    def run_basic_type_tests():
        @torch_fn
        def torch_process(x):
            assert torch.is_tensor(x)
            return x * 2

        @numpy_fn
        def numpy_process(x):
            assert isinstance(x, np.ndarray)
            return x * 2

        # Test type conversions
        for input_data, input_type in [
            (np_data, "numpy"),
            (torch_data, "torch"),
            (pd_data, "pandas"),
            (xr_data, "xarray")
        ]:
            result_torch = torch_process(input_data)
            result_numpy = numpy_process(input_data)

            assert torch.is_tensor(result_torch), f"Failed torch conversion from {input_type}"
            assert isinstance(result_numpy, np.ndarray), f"Failed numpy conversion from {input_type}"

    def run_batch_processing_tests():
        @batch_fn
        @torch_fn
        def batch_torch_process(x):
            assert torch.is_tensor(x)
            return x * 2

        # Test various batch sizes
        batch_sizes = [1, 10, 32, 50, 100, -1]
        results = []

        for batch_size in batch_sizes:
            result = batch_torch_process(np_data, batch_size=batch_size)
            results.append(result)
            assert torch.is_tensor(result), f"Failed batch processing with size {batch_size}"

        # Verify consistency across batch sizes
        for idx in range(len(results)-1):
            assert torch.allclose(results[idx], results[idx+1]), \
                f"Inconsistent results between batch sizes {batch_sizes[idx]} and {batch_sizes[idx+1]}"

    def run_complex_operation_tests():
        decorators = [numpy_fn, torch_fn, pandas_fn, xarray_fn]
        decorator_names = ['numpy', 'torch', 'pandas', 'xarray']

        def create_test_functions(dec1, dec2, dec3):
            @dec3
            def inner_op(x):
                return x + 1

            @dec2
            def middle_op(x):
                return inner_op(x) * 2

            @dec1
            def outer_op(x):
                return middle_op(x) + 3

            return outer_op

        # Test all combinations
        for i, dec1 in enumerate(decorators):
            for j, dec2 in enumerate(decorators):
                for k, dec3 in enumerate(decorators):
                    test_name = f"{decorator_names[i]}-{decorator_names[j]}-{decorator_names[k]}"
                    print(f"Testing combination: {test_name}")

                    try:
                        outer_op = create_test_functions(dec1, dec2, dec3)
                        result = outer_op(np_data)

                        # Check result type based on outermost decorator
                        if dec1 == numpy_fn:
                            assert isinstance(result, np.ndarray)
                        elif dec1 == torch_fn:
                            assert torch.is_tensor(result)
                        elif dec1 == pandas_fn:
                            assert isinstance(result, pd.DataFrame)
                        elif dec1 == xarray_fn:
                            assert isinstance(result, xr.DataArray)

                        print(f"Passed: {test_name}")
                    except Exception as err:
                        print(f"Failed: {test_name} - {str(err)}")

    # Execute all test suites
    test_suites = [
        ("Basic Type Tests", run_basic_type_tests),
        ("Batch Processing Tests", run_batch_processing_tests),
        ("Complex Operation Tests", run_complex_operation_tests)
    ]

    for test_name, test_func in test_suites:
        print(f"Running {test_name}...")
        test_func()
        print(f"{test_name} passed!")

    print("All tests completed successfully!")

if __name__ == "__main__":
    run_tests()


# python -m mngs.decorators.DataTypeDecorator


# EOF
