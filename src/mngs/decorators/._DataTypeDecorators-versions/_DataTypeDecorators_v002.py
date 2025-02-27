#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-12-05 09:34:05 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/_DataTypeDecorators.py

__file__ = "./src/mngs/decorators/_DataTypeDecorators.py"

"""
1. Functionality:
   - Provides decorators for data type conversion and batch processing
2. Input:
   - Functions to be decorated with data type and batch processing capabilities
3. Output:
   - Processed data with input type (regardless of the decorators) and batch handling; For example, if input arrays are numpy/pandas/torch/pandas, the output arrays are numpy/numpy/torch/numpy, respectively.
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
from ..types import is_array_like


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
    def restore_type(data: Any, original: Any) -> Any:
        if original is None or data is None:
            return data
        if isinstance(data, type(original)):
            return data

        # Move CUDA tensor to CPU if needed
        if torch.is_tensor(data) and data.device.type == 'cuda':
            data = data.cpu()

        # Convert to numpy first
        numpy_data = DataProcessor.convert_to_type(data, "numpy")

        # Then convert to the original type
        if isinstance(original, np.ndarray):
            return numpy_data
        elif isinstance(original, pd.DataFrame):
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
            tensor = torch.from_numpy(numpy_data)
            return tensor.to(original.device) if original.device.type == 'cuda' else tensor
        else:
            # For any other type, try direct conversion
            return type(original)(numpy_data)

    @staticmethod
    def convert_to_type(
        data: Any, target_type: str, device: str = None
    ) -> Any:
        if data is None:
            return None

        # Move CUDA tensor to CPU if needed for conversion
        if torch.is_tensor(data) and data.device.type == 'cuda':
            data = data.cpu()

        original_type = type(data)
        if isinstance(data, (int, float, bool, str)):
            return data

        device = device or DataProcessor._default_device

        # Handle scalar arrays explicitly
        if isinstance(data, (np.ndarray, torch.Tensor)):
            if data.size == 1:
                scalar_value = data.item()
                return original_type(
                    scalar_value
                )

        # Convert arrays preserving original attributes
        if target_type == "torch":
            if torch.is_tensor(data):
                return data.to(device)
            try:
                numpy_data = (
                    data.values
                    if isinstance(
                        data, (pd.Series, pd.DataFrame, xr.DataArray)
                    )
                    else (
                        data
                        if isinstance(data, np.ndarray)
                        else np.array(data, dtype=np.float64)
                    )
                )
                return torch.tensor(
                    numpy_data, device=device, dtype=torch.float64
                )
            except Exception as e:
                print(f"Error converting to torch: {e}")
                raise

        if target_type == "numpy":
            try:
                if isinstance(data, np.ndarray):
                    return data.astype(np.float64)
                if torch.is_tensor(data):
                    return data.cpu().numpy()
                if isinstance(data, (pd.Series, pd.DataFrame, xr.DataArray)):
                    return data.values
                return np.array(data, dtype=np.float64)
            except Exception as e:
                print(f"Error converting to numpy: {e}")
                raise

        return data

    @staticmethod
    def process_batches(
        func: Callable,
        data: Any,
        batch_size: int = -1,
        device: str = None,
    ) -> Any:

        if batch_size <= 0 or np.isscalar(data):
            return func(data)

        if not hasattr(data, "__len__"):
            return func(data)

        if batch_size == len(data):
            return func(data)

        original_type = type(data)
        total_size = len(data)
        results = []

        for i in range(0, total_size, batch_size):
            batch = data[i : i + batch_size]
            try:
                batch_result = func(batch)
                if np.isscalar(batch_result) or (
                    isinstance(batch_result, (np.ndarray, torch.Tensor))
                    and batch_result.size == 1
                ):
                    scalar_value = (
                        batch_result
                        if np.isscalar(batch_result)
                        else batch_result.item()
                    )
                    results.append(scalar_value)
                else:
                    results.append(batch_result)
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {str(e)}")
                raise

        if len(results) == 0:
            return None

        if all(np.isscalar(r) for r in results):
            result_array = np.array(results, dtype=np.float64)
            return DataProcessor.restore_type(
                result_array, data
            )  # Restore original type

        concatenated = np.concatenate(results, axis=0)
        return DataProcessor.restore_type(
            concatenated, data
        )  # Restore original type


def create_decorator(target_type: str = None, enable_batch: bool = False):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, device=None, **kwargs):
            def _convert_array(arr):
                if isinstance(arr, (np.ndarray, torch.Tensor)):
                    if isinstance(arr, np.ndarray):
                        return arr.astype(np.float64)
                    return arr.float()
                return arr

            batch_size = kwargs.pop("batch_size", -1) if enable_batch else -1

            is_method = args and hasattr(args[0], func.__name__)
            method_self = args[0] if is_method else None
            data_args = args[1:] if is_method else args

            # Store original data
            original_data = data_args[0] if data_args else None

            # Convert all array-like arguments
            if target_type:
                converted_args = [
                    (
                        _convert_array(
                            DataProcessor.convert_to_type(
                                arg, target_type, device
                            )
                        )
                        if is_array_like(arg)
                        else arg
                    )
                    for arg in data_args
                ]
                converted_kwargs = {
                    k: (
                        _convert_array(
                            DataProcessor.convert_to_type(
                                v, target_type, device
                            )
                        )
                        if is_array_like(v)
                        else v
                    )
                    for k, v in kwargs.items()
                }
            else:
                converted_args = [
                    _convert_array(arg) if is_array_like(arg) else arg
                    for arg in data_args
                ]
                converted_kwargs = {
                    k: _convert_array(v) if is_array_like(v) else v
                    for k, v in kwargs.items()
                }

            try:
                if enable_batch and original_data is not None:
                    if is_method:
                        result = DataProcessor.process_batches(
                            lambda x: func(
                                method_self,
                                x,
                                *converted_args[1:],
                                **converted_kwargs,
                            ),
                            converted_args[0],
                            batch_size,
                            device,
                        )
                    else:
                        result = DataProcessor.process_batches(
                            lambda x: func(
                                x, *converted_args[1:], **converted_kwargs
                            ),
                            converted_args[0],
                            batch_size,
                            device,
                        )
                else:
                    if is_method:
                        result = func(
                            method_self, *converted_args, **converted_kwargs
                        )
                    else:
                        result = func(*converted_args, **converted_kwargs)

                if original_data is not None:
                    if isinstance(result, (list, tuple)):
                        result = type(result)(
                            DataProcessor.restore_type(r, original_data)
                            for r in result
                        )
                    else:
                        result = DataProcessor.restore_type(
                            result, original_data
                        )

                return result
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                raise

        return wrapper

    return decorator


# Decorators
torch_fn = create_decorator(target_type="torch")
numpy_fn = create_decorator(target_type="numpy")
pandas_fn = create_decorator(target_type="pandas")
xarray_fn = create_decorator(target_type="xarray")
batch_fn = create_decorator(enable_batch=True)

# Combined batch + dtype decorators
batch_torch_fn = lambda func: batch_fn(torch_fn(func))
batch_numpy_fn = lambda func: batch_fn(numpy_fn(func))
batch_pandas_fn = lambda func: batch_fn(pandas_fn(func))
batch_xarray_fn = lambda func: batch_fn(xarray_fn(func))


def run_tests():
    import mngs

    # Test data creation with fixed seed
    np.random.seed(42)
    np_data = np.random.randn(100, 10)
    torch_data = torch.from_numpy(np_data)
    pd_data = pd.DataFrame(np_data, columns=[f"col_{i}" for i in range(10)])
    xr_data = xr.DataArray(np_data, dims=["samples", "features"])

    # def run_basic_type_tests():
    #     @torch_fn
    #     def torch_process(x):
    #         assert torch.is_tensor(x)
    #         return x * 2

    #     @numpy_fn
    #     def numpy_process(x):
    #         assert isinstance(x, np.ndarray)
    #         return x * 2

    #     # Test type conversions
    #     for input_data, input_type in [
    #         (np_data, "numpy"),
    #         (torch_data, "torch"),
    #         (pd_data, "pandas"),
    #         (xr_data, "xarray"),
    #     ]:
    #         result_torch = torch_process(input_data)
    #         result_numpy = numpy_process(input_data)

    #         assert torch.is_tensor(
    #             result_torch
    #         ), f"Failed torch conversion from {input_type}"
    #         assert isinstance(
    #             result_numpy, np.ndarray
    #         ), f"Failed numpy conversion from {input_type}"

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

            # Check that output type matches input type
            assert isinstance(
                result_torch, type(input_data)
            ), f"Failed: torch_process output type doesn't match input type {input_type}"
            assert isinstance(
                result_numpy, type(input_data)
            ), f"Failed: numpy_process output type doesn't match input type {input_type}"

            # mngs.str.printc(f"Passed: Type consistency for {input_type}", "green")

    # def run_combined_batch_tests():
    #     # Test combined batch decorators
    #     @batch_torch_fn
    #     def torch_batch_process(x):
    #         assert torch.is_tensor(x)
    #         return x * 2

    #     @batch_numpy_fn
    #     def numpy_batch_process(x):
    #         assert isinstance(x, np.ndarray)
    #         return x * 2

    #     @batch_pandas_fn
    #     def pandas_batch_process(x):
    #         assert isinstance(x, pd.DataFrame)
    #         return x * 2

    #     @batch_xarray_fn
    #     def xarray_batch_process(x):
    #         assert isinstance(x, xr.DataArray)
    #         return x * 2

    #     batch_sizes = [1, 10, 32, 50, 100, -1]
    #     for batch_size in batch_sizes:
    #         try:
    #             # Test torch batch
    #             result_torch = torch_batch_process(np_data, batch_size=batch_size)
    #             assert torch.is_tensor(result_torch)

    #             # Test numpy batch
    #             result_numpy = numpy_batch_process(np_data, batch_size=batch_size)
    #             assert isinstance(result_numpy, np.ndarray)

    #             # Test pandas batch
    #             result_pandas = pandas_batch_process(np_data, batch_size=batch_size)
    #             assert isinstance(result_pandas, pd.DataFrame)

    #             # Test xarray batch
    #             result_xarray = xarray_batch_process(np_data, batch_size=batch_size)
    #             assert isinstance(result_xarray, xr.DataArray)

    #             mngs.str.printc(f"Passed: Combined batch tests with batch_size={batch_size}", "yellow")
    #         except Exception as err:
    #             mngs.str.printc(f"Failed: Combined batch tests with batch_size={batch_size} - {str(err)}", "red")

    def run_batch_processing_tests():
        decorators = [torch_fn, numpy_fn, pandas_fn, xarray_fn]
        decorator_names = ["torch", "numpy", "pandas", "xarray"]

        for dtype_dec, dtype_name in zip(decorators, decorator_names):

            @dtype_dec
            @batch_fn
            def batch_first_process(x, factor=2):
                return x * factor

            test_name = f"{dtype_name}-dtype_first"
            try:
                test_data = np_data.copy()
                for batch_size in [1, 10, 32]:
                    result = batch_first_process(
                        test_data, factor=2, batch_size=batch_size
                    )
                    assert isinstance(result, type(test_data))
                mngs.str.printc(f"Passed: {test_name}", "yellow")
            except Exception as err:
                mngs.str.printc(f"Failed: {test_name} - {str(err)}", "red")

    def run_positional_args_tests():
        def run_basic_positional_tests():
            @batch_torch_fn
            def process_with_args(x, factor, param1, param2=2):
                return x * factor * param1 * param2

            try:
                test_data = np_data.copy()
                result = process_with_args(
                    test_data, 2, 3, param2=4, batch_size=32
                )
                assert isinstance(result, type(test_data))
                mngs.str.printc(
                    "Passed: Basic positional arguments tests", "yellow"
                )
            except Exception as err:
                mngs.str.printc(
                    f"Failed: Basic positional arguments tests - {str(err)}",
                    "red",
                )

        def run_method_positional_tests():
            class Processor:
                @batch_torch_fn
                def process_method(self, x, factor, param1, param2=2):
                    return x * factor * param1 * param2

            try:
                processor = Processor()
                test_data = np_data.copy()
                result = processor.process_method(
                    test_data, 2, 3, param2=4, batch_size=32
                )
                assert isinstance(result, type(test_data))
                mngs.str.printc(
                    "Passed: Method positional arguments tests", "yellow"
                )
            except Exception as err:
                mngs.str.printc(
                    f"Failed: Method positional arguments tests - {str(err)}",
                    "red",
                )

        run_basic_positional_tests()
        run_method_positional_tests()

    def run_combined_batch_tests():
        # Test combined batch decorators
        @batch_torch_fn
        def torch_batch_process(x):
            return x * 2

        @batch_numpy_fn
        def numpy_batch_process(x):
            return x * 2

        @batch_pandas_fn
        def pandas_batch_process(x):
            return x * 2

        @batch_xarray_fn
        def xarray_batch_process(x):
            return x * 2

        batch_sizes = [1, 10, 32, 50, 100, -1]
        for batch_size in batch_sizes:
            try:
                # Test with numpy array input
                test_data = np_data.copy()

                result_torch = torch_batch_process(
                    test_data, batch_size=batch_size
                )
                assert isinstance(result_torch, np.ndarray)

                result_numpy = numpy_batch_process(
                    test_data, batch_size=batch_size
                )
                assert isinstance(result_numpy, np.ndarray)

                result_pandas = pandas_batch_process(
                    test_data, batch_size=batch_size
                )
                assert isinstance(result_pandas, np.ndarray)

                result_xarray = xarray_batch_process(
                    test_data, batch_size=batch_size
                )
                assert isinstance(result_xarray, np.ndarray)

                # mngs.str.printc(f"Passed: Combined batch tests with batch_size={batch_size}", "yellow")
            except Exception as err:
                mngs.str.printc(
                    f"Failed: Combined batch tests with batch_size={batch_size} - {str(err)}",
                    "red",
                )

    def run_cuda_tests():
        if not torch.cuda.is_available():
            mngs.str.printc("CUDA tests skipped: No GPU available", "yellow")
            return

        @torch_fn
        def auto_device_fn(x):
            if torch.is_tensor(x):
                assert x.device.type == "cuda"
            return x * 2

        @torch_fn
        def explicit_device_fn(x):
            return x * 2

        try:
            test_data = torch.tensor(np.random.randn(100, 10))

            result_auto = auto_device_fn(test_data)
            assert isinstance(result_auto, torch.Tensor)

            result_cuda = explicit_device_fn(test_data, device="cuda")
            result_cpu = explicit_device_fn(test_data, device="cpu")
            assert isinstance(result_cuda, torch.Tensor)
            assert isinstance(result_cpu, torch.Tensor)

            @batch_fn
            @torch_fn
            def batch_cuda_fn(x):
                if torch.is_tensor(x):
                    assert x.device.type == "cuda"
                return x * 2

            result_batch = batch_cuda_fn(test_data, batch_size=32)
            assert isinstance(result_batch, torch.Tensor)

            # mngs.str.printc("Passed: CUDA tests", "yellow")
        except Exception as err:
            mngs.str.printc(f"Failed: CUDA tests - {str(err)}", "red")

    def run_nested_operation_tests():
        decorators = [numpy_fn, torch_fn, pandas_fn, xarray_fn]
        decorator_names = ["numpy", "torch", "pandas", "xarray"]

        def create_test_functions(dec1, dec2, dec3):
            @dec3
            def inner_op(x):
                return x + 1

            @dec2
            def middle_op(x):
                inner_result = inner_op(x)
                # Ensure type consistency after inner operation
                if isinstance(x, type(inner_result)):
                    return inner_result * 2
                return DataProcessor.restore_type(inner_result * 2, x)

            @dec1
            def outer_op(x):
                middle_result = middle_op(x)
                # Ensure type consistency after middle operation
                if isinstance(x, type(middle_result)):
                    return middle_result + 3
                return DataProcessor.restore_type(middle_result + 3, x)

            return outer_op

        # Test all combinations
        for i, dec1 in enumerate(decorators):
            for j, dec2 in enumerate(decorators):
                for k, dec3 in enumerate(decorators):
                    test_name = f"{decorator_names[i]}-{decorator_names[j]}-{decorator_names[k]}"
                    try:
                        outer_op = create_test_functions(dec1, dec2, dec3)
                        result = outer_op(np_data)
                        assert isinstance(
                            result, type(np_data)
                        ), f"Type mismatch: expected {type(np_data)}, got {type(result)}"
                        # mngs.str.printc(f"Passed: {test_name}", "yellow")
                    except Exception as err:
                        mngs.str.printc(
                            f"Failed: {test_name} - {str(err)}", "red"
                        )

    # def run_cuda_tests():
    #     if not torch.cuda.is_available():
    #         mngs.str.printc("CUDA tests skipped: No GPU available", "yellow")
    #         return

    #     # Test automatic device selection
    #     @torch_fn
    #     def auto_device_fn(x):
    #         assert x.device.type == "cuda"
    #         return x * 2

    #     # Test explicit device selection
    #     @torch_fn
    #     def explicit_device_fn(x):
    #         return x * 2

    #     # Test data
    #     test_data = np.random.randn(100, 10)

    #     try:
    #         # Test auto device selection
    #         result_auto = auto_device_fn(test_data)
    #         assert result_auto.device.type == "cuda"
    #         mngs.str.printc("Passed: Auto CUDA device selection", "yellow")

    #         # Test explicit device selection
    #         result_cuda = explicit_device_fn(test_data, device="cuda")
    #         result_cpu = explicit_device_fn(test_data, device="cpu")
    #         assert result_cuda.device.type == "cuda"
    #         assert result_cpu.device.type == "cpu"
    #         mngs.str.printc("Passed: Explicit device selection", "yellow")

    #         # Test batch processing with CUDA
    #         @batch_fn
    #         @torch_fn
    #         def batch_cuda_fn(x):
    #             assert x.device.type == "cuda"
    #             return x * 2

    #         result_batch = batch_cuda_fn(test_data, batch_size=32)
    #         assert result_batch.device.type == "cuda"
    #         mngs.str.printc("Passed: Batch processing with CUDA", "yellow")

    #     except Exception as err:
    #         mngs.str.printc(f"Failed: CUDA tests - {str(err)}", "red")

    # def run_positional_args_tests():
    #     import mngs

    #     def run_basic_positional_tests():
    #         @batch_torch_fn
    #         def process_with_args(x, fs, param1, param2=2):
    #             assert torch.is_tensor(x)
    #             return x * fs * param1 * param2

    #         @torch_fn
    #         def simple_with_args(x, y, z):
    #             assert torch.is_tensor(x)
    #             assert torch.is_tensor(y)
    #             assert torch.is_tensor(z)
    #             return x * y + z

    #         try:
    #             # Test batch processing with additional args
    #             result = process_with_args(np_data, 512, 2, param2=3, batch_size=32)
    #             assert torch.is_tensor(result)

    #             # Test multiple positional args
    #             result = simple_with_args(np_data, np_data * 2, np_data + 1)
    #             assert torch.is_tensor(result)

    #             # mngs.str.printc("Passed: Basic positional arguments tests", "yellow")
    #         except Exception as err:
    #             mngs.str.printc(f"Failed: Basic positional arguments tests - {str(err)}", "red")

    #     def run_method_positional_tests():
    #         class Processor:
    #             @batch_torch_fn
    #             def process_with_args(self, x, fs, param1, param2=2):
    #                 assert torch.is_tensor(x)
    #                 return x * fs * param1 * param2

    #             @torch_fn
    #             def simple_with_args(self, x, y, z):
    #                 assert torch.is_tensor(x)
    #                 assert torch.is_tensor(y)
    #                 assert torch.is_tensor(z)
    #                 return x * y + z

    #         processor = Processor()
    #         try:
    #             # Test batch processing with additional args
    #             result = processor.process_with_args(np_data, 512, 2, param2=3, batch_size=32)
    #             assert torch.is_tensor(result)

    #             # Test multiple positional args
    #             result = processor.simple_with_args(np_data, np_data * 2, np_data + 1)
    #             assert torch.is_tensor(result)

    #             # mngs.str.printc("Passed: Method positional arguments tests", "yellow")
    #         except Exception as err:
    #             mngs.str.printc(f"Failed: Method positional arguments tests - {str(err)}", "red")

    #     test_suites = [
    #         ("Basic Positional Tests", run_basic_positional_tests),
    #         ("Method Positional Tests", run_method_positional_tests),
    #     ]

    #     for test_name, test_func in test_suites:
    #         test_func()

    def run_datatype_consistency_tests():
        test_cases = [
            (np_data, np.ndarray),
            (torch_data, torch.Tensor),
            (pd_data, pd.DataFrame),
            (xr_data, xr.DataArray),
        ]

        decorators = [
            (torch_fn, "torch_fn"),
            (numpy_fn, "numpy_fn"),
            (pandas_fn, "pandas_fn"),
            (xarray_fn, "xarray_fn"),
            (batch_torch_fn, "batch_torch_fn"),
            (batch_numpy_fn, "batch_numpy_fn"),
            (batch_pandas_fn, "batch_pandas_fn"),
            (batch_xarray_fn, "batch_xarray_fn"),
        ]

        for dec, dec_name in decorators:
            for input_data, expected_type in test_cases:

                @dec
                def process(x):
                    return x * 2

                try:
                    result = process(input_data)
                    assert isinstance(
                        result, type(input_data)
                    ), f"Type mismatch with {dec_name}: input {type(input_data)} -> output {type(result)}"
                    # mngs.str.printc(f"Passed: {dec_name} maintains {type(input_data)} type", "green")
                except Exception as e:
                    mngs.str.printc(
                        f"Failed: {dec_name} with {type(input_data)} - {str(e)}",
                        "red",
                    )

    # Execute all test suites
    test_suites = [
        ("Basic Type Tests", run_basic_type_tests),
        ("Batch Processing Tests", run_batch_processing_tests),
        ("Nested Operation Tests", run_nested_operation_tests),
        ("Combined Batch Tests", run_combined_batch_tests),
        ("CUDA Tests", run_cuda_tests),
        ("Positional Arguments Tests", run_positional_args_tests),
        ("Datatype Consistency Tests", run_datatype_consistency_tests),
        # ("Scalar and Device Tests", run_scalar_device_tests),
    ]

    for test_name, test_func in test_suites:
        # print(f"Running {test_name}...")
        test_func()
        # mngs.str.printc(f"{test_name} passed!", "yellow")
        # print()

    mngs.str.printc("All tests completed successfully!", "yellow")


if __name__ == "__main__":
    run_tests()


# python -m mngs.decorators._DataTypeDecorators


# EOF



"""
python ./mngs_repo/src/mngs/decorators/_DataTypeDecorators.py
python -m src.mngs.decorators._DataTypeDecorators
"""
# (.env-3.11-home) (SpartanGPU) mngs_repo $ python -m src.mngs.decorators._DataTypeDecorators
# <frozen runpy>:128: RuntimeWarning: 'src.mngs.decorators._DataTypeDecorators' found in sys.modules after import of package 'src.mngs.decorators', but prior to execution of 'src.mngs.decorators._DataTypeDecorators'; this may result in unpredictable behaviour
# Error processing batch: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
# Error processing batch: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

# ----------------------------------------
# Failed: torch-dtype_first - can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
# ----------------------------------------


# ----------------------------------------
# Passed: numpy-dtype_first
# ----------------------------------------


# ----------------------------------------
# Passed: pandas-dtype_first
# ----------------------------------------


# ----------------------------------------
# Passed: xarray-dtype_first
# ----------------------------------------

# Error processing batch: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

# ----------------------------------------
# Failed: CUDA tests - can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
# ----------------------------------------


# ----------------------------------------
# Passed: Basic positional arguments tests
# ----------------------------------------


# ----------------------------------------
# Passed: Method positional arguments tests
# ----------------------------------------


# ----------------------------------------
# All tests completed successfully!
# ----------------------------------------

# (.env-3.11-home) (SpartanGPU) mngs_repo $


"""
python ./mngs_repo/src/mngs/decorators/_DataTypeDecorators.py
python -m src.mngs.decorators._DataTypeDecorators
"""

# EOF
