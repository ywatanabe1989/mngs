# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-12-05 10:35:03 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_DataTypeDecorators.py
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_DataTypeDecorators.py"
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-12-05 09:44:14 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_DataTypeDecorators.py
# 
# __file__ = (
#     "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_DataTypeDecorators.py"
# )
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-12-05 09:40:26 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_DataTypeDecorators.py
# 
# __file__ = (
#     "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_DataTypeDecorators.py"
# )
# 
# """
# 1. Functionality:
#    - Provides decorators for data type conversion and batch processing
# 2. Input:
#    - Functions to be decorated with data type and batch processing capabilities
# 3. Output:
#    - Processed data with input type (regardless of the decorators) and batch handling; For example, if input arrays are numpy/pandas/torch/pandas, the output arrays are numpy/numpy/torch/numpy, respectively.
# 4. Prerequisites:
#    - torch, numpy, pandas, xarray
# """
# 
# """Imports"""
# import threading
# from contextlib import contextmanager
# from functools import wraps
# from typing import Any, Callable
# 
# import numpy as np
# import pandas as pd
# import torch
# import xarray as xr
# from mngs.str import printc
# 
# from ..types import is_array_like
# 
# 
# class DataProcessor:
#     _local = threading.local()
#     _default_device = "cuda" if torch.cuda.is_available() else "cpu"
# 
#     @staticmethod
#     def set_default_device(device: str):
#         DataProcessor._default_device = device
# 
#     @staticmethod
#     @contextmanager
#     def computation_context(dtype: str):
#         if not hasattr(DataProcessor._local, "dtype_stack"):
#             DataProcessor._local.dtype_stack = []
#         DataProcessor._local.dtype_stack.append(dtype)
#         try:
#             yield
#         finally:
#             DataProcessor._local.dtype_stack.pop()
# 
#     @staticmethod
#     def _convert_kwargs(kwargs: dict, target_type: str) -> dict:
#         converted = {}
#         for k, v in kwargs.items():
#             if k == "axis" and target_type == "torch":
#                 converted["dim"] = v
#             elif k == "dim" and target_type == "numpy":
#                 converted["axis"] = v
#             else:
#                 converted[k] = v
#         return converted
# 
#     @staticmethod
#     def restore_type(data: Any, original: Any) -> Any:
#         if original is None or data is None:
#             return data
#         if isinstance(data, type(original)):
#             return data
# 
#         # Move CUDA tensor to CPU if needed
#         if torch.is_tensor(data) and data.device.type == "cuda":
#             data = data.cpu()
# 
#         # Convert to numpy first
#         numpy_data = DataProcessor.convert_to_type(data, "numpy")
# 
#         # Then convert to the original type
#         if isinstance(original, np.ndarray):
#             return numpy_data
#         elif isinstance(original, pd.DataFrame):
#             return pd.DataFrame(
#                 numpy_data, index=original.index, columns=original.columns
#             )
#         elif isinstance(original, pd.Series):
#             return pd.Series(numpy_data, index=original.index)
#         elif isinstance(original, xr.DataArray):
#             return xr.DataArray(
#                 numpy_data, coords=original.coords, dims=original.dims
#             )
#         elif isinstance(original, torch.Tensor):
#             tensor = torch.from_numpy(numpy_data)
#             return (
#                 tensor.to(original.device)
#                 if original.device.type == "cuda"
#                 else tensor
#             )
#         else:
#             # For any other type, try direct conversion
#             return type(original)(numpy_data)
# 
#     @staticmethod
#     def convert_to_type(
#         data: Any, target_type: str, device: str = None
#     ) -> Any:
#         if data is None:
#             return None
# 
#         # Move CUDA tensor to CPU if needed for conversion
#         if torch.is_tensor(data) and data.device.type == "cuda":
#             data = data.cpu()
# 
#         original_type = type(data)
#         if isinstance(data, (int, float, bool, str)):
#             return data
# 
#         device = device or DataProcessor._default_device
# 
#         # Handle scalar arrays explicitly
#         if isinstance(data, (np.ndarray, torch.Tensor)):
#             if data.size == 1:
#                 scalar_value = data.item()
#                 return original_type(scalar_value)
# 
#         # Convert arrays preserving original attributes
#         if target_type == "torch":
#             if torch.is_tensor(data):
#                 return data.to(device)
#             try:
#                 numpy_data = (
#                     data.values
#                     if isinstance(
#                         data, (pd.Series, pd.DataFrame, xr.DataArray)
#                     )
#                     else (
#                         data
#                         if isinstance(data, np.ndarray)
#                         else np.array(data, dtype=np.float64)
#                     )
#                 )
#                 return torch.tensor(
#                     numpy_data, device=device, dtype=torch.float64
#                 )
#             except Exception as e:
#                 print(f"Error converting to torch: {e}")
#                 raise
# 
#         if target_type == "numpy":
#             try:
#                 if isinstance(data, np.ndarray):
#                     return data.astype(np.float64)
#                 if torch.is_tensor(data):
#                     return data.cpu().numpy()
#                 if isinstance(data, (pd.Series, pd.DataFrame, xr.DataArray)):
#                     return data.values
#                 return np.array(data, dtype=np.float64)
#             except Exception as e:
#                 print(f"Error converting to numpy: {e}")
#                 raise
# 
#         return data
# 
#     @staticmethod
#     def process_batches(
#         func: Callable,
#         data: Any,
#         batch_size: int = -1,
#         device: str = None,
#     ) -> Any:
# 
#         if batch_size <= 0 or np.isscalar(data):
#             return func(data)
# 
#         if not hasattr(data, "__len__"):
#             return func(data)
# 
#         if batch_size == len(data):
#             return func(data)
# 
#         original_type = type(data)
#         total_size = len(data)
#         results = []
# 
#         for i in range(0, total_size, batch_size):
#             batch = data[i : i + batch_size]
#             try:
#                 batch_result = func(batch)
#                 if np.isscalar(batch_result) or (
#                     isinstance(batch_result, (np.ndarray, torch.Tensor))
#                     and batch_result.size == 1
#                 ):
#                     scalar_value = (
#                         batch_result
#                         if np.isscalar(batch_result)
#                         else batch_result.item()
#                     )
#                     results.append(scalar_value)
#                 else:
#                     results.append(batch_result)
#             except Exception as e:
#                 print(f"Error processing batch {i//batch_size}: {str(e)}")
#                 raise
# 
#         if len(results) == 0:
#             return None
# 
#         if all(np.isscalar(r) for r in results):
#             result_array = np.array(results, dtype=np.float64)
#             return DataProcessor.restore_type(
#                 result_array, data
#             )  # Restore original type
# 
#         concatenated = np.concatenate(results, axis=0)
#         return DataProcessor.restore_type(
#             concatenated, data
#         )  # Restore original type
# 
# 
# def create_decorator(target_type: str = None, enable_batch: bool = False):
#     def decorator(func: Callable) -> Callable:
#         @wraps(func)
#         def wrapper(*args, device=None, **kwargs):
#             def _convert_array(arr):
#                 if isinstance(arr, (np.ndarray, torch.Tensor)):
#                     if isinstance(arr, np.ndarray):
#                         return arr.astype(np.float64)
#                     return arr.float()
#                 return arr
# 
#             batch_size = kwargs.pop("batch_size", -1) if enable_batch else -1
# 
#             is_method = args and hasattr(args[0], func.__name__)
#             method_self = args[0] if is_method else None
#             data_args = args[1:] if is_method else args
# 
#             # Store original data
#             original_data = data_args[0] if data_args else None
# 
#             # Convert all array-like arguments
#             if target_type:
#                 converted_args = [
#                     (
#                         _convert_array(
#                             DataProcessor.convert_to_type(
#                                 arg, target_type, device
#                             )
#                         )
#                         if is_array_like(arg)
#                         else arg
#                     )
#                     for arg in data_args
#                 ]
#                 converted_kwargs = {
#                     k: (
#                         _convert_array(
#                             DataProcessor.convert_to_type(
#                                 v, target_type, device
#                             )
#                         )
#                         if is_array_like(v)
#                         else v
#                     )
#                     for k, v in kwargs.items()
#                 }
#             else:
#                 converted_args = [
#                     _convert_array(arg) if is_array_like(arg) else arg
#                     for arg in data_args
#                 ]
#                 converted_kwargs = {
#                     k: _convert_array(v) if is_array_like(v) else v
#                     for k, v in kwargs.items()
#                 }
# 
#             try:
#                 if enable_batch and original_data is not None:
#                     if is_method:
#                         result = DataProcessor.process_batches(
#                             lambda x: func(
#                                 method_self,
#                                 x,
#                                 *converted_args[1:],
#                                 **converted_kwargs,
#                             ),
#                             converted_args[0],
#                             batch_size,
#                             device,
#                         )
#                     else:
#                         result = DataProcessor.process_batches(
#                             lambda x: func(
#                                 x, *converted_args[1:], **converted_kwargs
#                             ),
#                             converted_args[0],
#                             batch_size,
#                             device,
#                         )
#                 else:
#                     if is_method:
#                         result = func(
#                             method_self, *converted_args, **converted_kwargs
#                         )
#                     else:
#                         result = func(*converted_args, **converted_kwargs)
# 
#                 if original_data is not None:
#                     if isinstance(result, (list, tuple)):
#                         result = type(result)(
#                             DataProcessor.restore_type(r, original_data)
#                             for r in result
#                         )
#                     else:
#                         result = DataProcessor.restore_type(
#                             result, original_data
#                         )
# 
#                 return result
#             except Exception as e:
#                 print(f"Error processing batch: {str(e)}")
#                 raise
# 
#         return wrapper
# 
#     return decorator
# 
# 
# # Decorators
# torch_fn = create_decorator(target_type="torch")
# numpy_fn = create_decorator(target_type="numpy")
# pandas_fn = create_decorator(target_type="pandas")
# xarray_fn = create_decorator(target_type="xarray")
# batch_fn = create_decorator(enable_batch=True)
# 
# # Combined batch + dtype decorators
# batch_torch_fn = lambda func: batch_fn(torch_fn(func))
# batch_numpy_fn = lambda func: batch_fn(numpy_fn(func))
# batch_pandas_fn = lambda func: batch_fn(pandas_fn(func))
# batch_xarray_fn = lambda func: batch_fn(xarray_fn(func))
# 
# 
# """
# python src/mngs/decorators/_DataTypeDecorators.py
# python -m src.mngs.decorators._DataTypeDecorators
# """
# 
# # EOF

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from ...src.mngs..decorators._DataTypeDecorators import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        pass

    def test_edge_cases(self):
        # Edge case testing
        pass

    def test_error_handling(self):
        # Error handling testing
        pass
