# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:56:44 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_batch_fn.py
# 
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# 
# import torch
# from tqdm import tqdm as _tqdm
# 
# from ._converters import (_conversion_warning, _try_device, is_torch, to_numpy,
#                           to_torch)
# 
# 
# def batch_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(x: _Any, *args: _Any, **kwargs: _Any) -> _Any:
#         batch_size = int(kwargs.pop("batch_size", 4))
#         if len(x) <= batch_size:
#             return func(x, *args, **kwargs, batch_size=batch_size)
#         n_batches = (len(x) + batch_size - 1) // batch_size
#         results = []
#         for i_batch in _tqdm(range(n_batches)):
#             start = i_batch * batch_size
#             end = min((i_batch + 1) * batch_size, len(x))
#             batch_result = func(x[start:end], *args, **kwargs, batch_size=batch_size)
#             if isinstance(batch_result, torch.Tensor):
#                 batch_result = batch_result.cpu()
#             elif isinstance(batch_result, tuple):
#                 batch_result = tuple(
#                     val.cpu() if isinstance(val, torch.Tensor) else val
#                     for val in batch_result
#                 )
#             results.append(batch_result)
#         if isinstance(results[0], tuple):
#             n_vars = len(results[0])
#             combined_results = [
#                 torch.vstack([res[i_var] for res in results]) for i_var in range(n_vars)
#             ]
#             return tuple(combined_results)
#         elif isinstance(results[0], torch.Tensor):
#             return torch.vstack(results)
#         else:
#             return sum(results, [])
# 
#     return wrapper
# 
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

from mngs..decorators._batch_fn import *

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
