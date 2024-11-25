#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-25 20:58:23 (ywatanabe)"
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
    _default_device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def set_default_device(device: str):
        DataProcessor._default_device = device

    @staticmethod
    @contextmanager
    def computation_context(dtype: str):
        if not hasattr(DataProcessor._local, "dtype_stack"):
            DataProcessor._local.dtype_stack = []
        DataProcessor._local.dtype_stack.append(dtype)
        try:
            yield
        finally:
            DataProcessor._local.dtype_stack.pop()

    @staticmethod
    def _convert_kwargs(kwargs: dict, target_type: str) -> dict:
        converted = {}
        for k, v in kwargs.items():
            if k == "axis" and target_type == "torch":
                converted["dim"] = v
            elif k == "dim" and target_type == "numpy":
                converted["axis"] = v
            else:
                converted[k] = v
        return converted

    @staticmethod
    def convert_to_type(
        data: Any, target_type: str, device: str = None
    ) -> Any:
        if data is None:
            return None

        device = device or DataProcessor._default_device

        if isinstance(data, dict):
            return DataProcessor._convert_kwargs(data, target_type)

        if target_type == "torch":
            if torch.is_tensor(data):
                return data.to(device)
            numpy_data = (
                data.values
                if isinstance(data, (pd.Series, pd.DataFrame, xr.DataArray))
                else data if isinstance(data, np.ndarray) else np.array(data)
            )
            return torch.tensor(numpy_data, device=device)

        if torch.is_tensor(data):
            data = data.cpu().numpy()

        if target_type == "numpy":
            return (
                data.values
                if isinstance(data, (pd.Series, pd.DataFrame, xr.DataArray))
                else data
            )
        elif target_type == "pandas":
            if isinstance(data, pd.DataFrame):
                return data
            return pd.DataFrame(data)
        elif target_type == "xarray":
            if isinstance(data, xr.DataArray):
                return data
            return xr.DataArray(data)
        return data

    @staticmethod
    def restore_type(data: Any, original: Any) -> Any:
        if original is None or data is None:
            return data
        if isinstance(original, type(data)):
            return data

        numpy_data = DataProcessor.convert_to_type(data, "numpy")

        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(
                numpy_data, index=original.index, columns=original.columns
            )
        elif isinstance(original, pd.Series):
            return pd.Series(numpy_data, index=original.index)
        elif isinstance(original, xr.DataArray):
            return xr.DataArray(
                numpy_data, coords=original.coords, dims=original.dims
            )
        elif isinstance(original, torch.Tensor):
            return torch.from_numpy(numpy_data).to(original.device)
        return data

    @staticmethod
    def process_batches(
        func: Callable,
        data: Any,
        batch_size: Union[int, Tuple[int]],
        device: str = None,
        **kwargs,
    ) -> Any:
        device = device or DataProcessor._default_device

        if batch_size == -1:
            return func(data, **kwargs)

        if isinstance(batch_size, int):
            batch_size = (batch_size,)

        total_size = data.shape[0]
        batches = [
            data[i : i + batch_size[0]]
            for i in range(0, total_size, batch_size[0])
        ]
        results = [func(batch, **kwargs) for batch in batches]

        if len(results) == 0:
            return None

        if torch.is_tensor(results[0]):
            return torch.cat(results, dim=0)
        elif isinstance(results[0], np.ndarray):
            return np.concatenate(results, axis=0)
        elif isinstance(results[0], pd.DataFrame):
            return pd.concat(results)
        elif isinstance(results[0], xr.DataArray):
            return xr.concat(results, dim=results[0].dims[0])
        return results


# working for batch_fn first decoration
def create_decorator(target_type: str = None, enable_batch: bool = False):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, device=None, **kwargs):
            batch_size = kwargs.pop("batch_size", -1) if enable_batch else -1

            current_type = (
                DataProcessor._local.dtype_stack[-1]
                if hasattr(DataProcessor._local, "dtype_stack")
                and DataProcessor._local.dtype_stack
                else target_type
            )

            if current_type:
                converted_args = [
                    DataProcessor.convert_to_type(arg, current_type, device)
                    for arg in args
                ]
                converted_kwargs = {
                    k: DataProcessor.convert_to_type(v, current_type, device)
                    for k, v in kwargs.items()
                }
            else:
                converted_args = args
                converted_kwargs = kwargs

            if enable_batch:
                result = DataProcessor.process_batches(
                    lambda x: func(x, **converted_kwargs),
                    converted_args[0],
                    batch_size,
                    device,
                )
            else:
                result = func(*converted_args, **converted_kwargs)

            if target_type:
                result = DataProcessor.convert_to_type(
                    result, target_type, device
                )

            return result

        return wrapper

    return decorator


# Decorators
torch_fn = create_decorator(target_type="torch")
numpy_fn = create_decorator(target_type="numpy")
pandas_fn = create_decorator(target_type="pandas")
xarray_fn = create_decorator(target_type="xarray")
batch_fn = create_decorator(enable_batch=True)


def run_tests():
    import mngs

    # Test data creation with fixed seed
    np.random.seed(42)
    np_data = np.random.randn(100, 10)
    torch_data = torch.from_numpy(np_data)
    pd_data = pd.DataFrame(np_data, columns=[f"col_{i}" for i in range(10)])
    xr_data = xr.DataArray(np_data, dims=["samples", "features"])

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
            (xr_data, "xarray"),
        ]:
            result_torch = torch_process(input_data)
            result_numpy = numpy_process(input_data)

            assert torch.is_tensor(
                result_torch
            ), f"Failed torch conversion from {input_type}"
            assert isinstance(
                result_numpy, np.ndarray
            ), f"Failed numpy conversion from {input_type}"


    def run_batch_processing_tests():
        decorators = [torch_fn, numpy_fn, pandas_fn, xarray_fn]
        decorator_names = ["torch", "numpy", "pandas", "xarray"]

        # Test both decorator orders
        for dtype_dec, dtype_name in zip(decorators, decorator_names):
            # Test dtype_fn first (correct order)
            @batch_fn
            @dtype_dec
            def batch_first_process(x):
                if dtype_name == "torch":
                    assert torch.is_tensor(x)
                elif dtype_name == "numpy":
                    assert isinstance(x, np.ndarray)
                elif dtype_name == "pandas":
                    assert isinstance(x, pd.DataFrame)
                elif dtype_name == "xarray":
                    assert isinstance(x, xr.DataArray)
                return x * 2

            test_name = f"{dtype_name}-dtype_first"
            try:
                batch_sizes = [1, 10, 32, 50, 100, -1]
                results = []

                # Create appropriate test data based on type
                if dtype_name == "pandas":
                    test_data = pd.DataFrame(
                        np_data,
                        columns=[f"col_{i}" for i in range(np_data.shape[1])],
                    )
                elif dtype_name == "xarray":
                    test_data = xr.DataArray(
                        np_data, dims=["samples", "features"]
                    )
                else:
                    test_data = np_data

                for batch_size in batch_sizes:
                    result = batch_first_process(test_data, batch_size=batch_size)
                    results.append(result)

                    # Verify type
                    if dtype_name == "torch":
                        assert torch.is_tensor(result)
                    elif dtype_name == "numpy":
                        assert isinstance(result, np.ndarray)
                    elif dtype_name == "pandas":
                        assert isinstance(result, pd.DataFrame)
                    elif dtype_name == "xarray":
                        assert isinstance(result, xr.DataArray)

                # Verify consistency across batch sizes
                for idx in range(len(results) - 1):
                    if dtype_name == "torch":
                        assert torch.allclose(results[idx], results[idx + 1])
                    elif dtype_name == "numpy":
                        assert np.allclose(results[idx], results[idx + 1])
                    elif dtype_name == "pandas":
                        assert np.allclose(results[idx].values, results[idx + 1].values)
                    elif dtype_name == "xarray":
                        assert np.allclose(results[idx].values, results[idx + 1].values)

                mngs.str.printc(f"Passed: {test_name}", "yellow")
            except Exception as err:
                mngs.str.printc(f"Failed: {test_name} - {str(err)}", "red")

    def run_complex_operation_tests():
        decorators = [numpy_fn, torch_fn, pandas_fn, xarray_fn]
        decorator_names = ["numpy", "torch", "pandas", "xarray"]

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
                    # print(f"Testing combination: {test_name}")

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

                        # mngs.str.printc(f"Passed: {test_name}", "yellow")
                    except Exception as err:
                        mngs.str.printc(
                            f"Failed: {test_name} - {str(err)}", "red"
                        )

    # Execute all test suites
    test_suites = [
        ("Basic Type Tests", run_basic_type_tests),
        ("Batch Processing Tests", run_batch_processing_tests),
        ("Complex Operation Tests", run_complex_operation_tests),
    ]

    for test_name, test_func in test_suites:
        # print(f"Running {test_name}...")
        test_func()
        # mngs.str.printc(f"{test_name} passed!", "yellow")
        # print()

    mngs.str.printc("All tests completed successfully!", "yellow")


if __name__ == "__main__":
    run_tests()


# python -m mngs.decorators.DataTypeDecorator


# EOF
