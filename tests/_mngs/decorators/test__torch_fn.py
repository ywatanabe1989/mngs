# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-06 15:45:12 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_torch_fn.py
# 
# """
# Functionality:
#     - Implements PyTorch-specific conversion and utility functions
#     - Provides decorators for PyTorch operations
# Input:
#     - Various data types to be converted to PyTorch tensors
# Output:
#     - PyTorch tensors and processing results
# Prerequisites:
#     - PyTorch package
#     - Core converter utilities
# """
# 
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# 
# import numpy as np
# import pandas as pd
# import torch
# 
# from ._converters import (
#     _conversion_warning,
#     _return_always,
#     _return_if,
#     is_torch,
#     to_numpy,
#     to_torch,
# )
# 
# 
# # def torch_fn(func: Callable) -> Callable:
# #     @wraps(func)
# #     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
# #         is_torch_input = is_torch(*args, **kwargs)
# #         converted_args, converted_kwargs = to_torch(
# #             *args, return_fn=_return_always, **kwargs
# #         )
# #         results = func(*converted_args, **converted_kwargs)
# #         return (
# #             to_numpy(results, return_fn=_return_if)[0]
# #             if not is_torch_input
# #             else results
# #         )
# 
# #     return wrapper
# 
# def torch_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         # Skip conversion if inputs are already torch tensors
#         if is_torch_input:
#             results = func(*args, **kwargs)
#         else:
#             converted_args, converted_kwargs = to_torch(
#                 *args, return_fn=_return_always, **kwargs
#             )
#             results = func(*converted_args, **converted_kwargs)
#             results = to_numpy(results, return_fn=_return_if)[0]
#         return results
# 
#     return wrapper
# 
# 
# if __name__ == "__main__":
#     import scipy
#     import torch.nn.functional as F
# 
#     @torch_fn
#     def torch_softmax(*args: _Any, **kwargs: _Any) -> torch.Tensor:
#         return F.softmax(*args, **kwargs)
# 
#     def custom_print(data: _Any) -> None:
#         print(type(data), data)
# 
#     test_data = [1, 2, 3]
#     test_list = test_data
#     test_tensor = torch.tensor(test_data).float()
#     test_tensor_cuda = torch.tensor(test_data).float().cuda()
#     test_array = np.array(test_data)
#     test_df = pd.DataFrame({"col1": test_data})
# 
#     print("Testing torch_fn:")
#     custom_print(torch_softmax(test_list, dim=-1))
#     custom_print(torch_softmax(test_array, dim=-1))
#     custom_print(torch_softmax(test_df, dim=-1))
#     custom_print(torch_softmax(test_tensor, dim=-1))
#     custom_print(torch_softmax(test_tensor_cuda, dim=-1))
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

from mngs..decorators._torch_fn import *

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
