# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-26 18:46:08 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_pandas_fn.py
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py"
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:55:46 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_pandas_fn.py
# 
# """
# 1. Functionality:
#    - (e.g., Executes XYZ operation)
# 2. Input:
#    - (e.g., Required data for XYZ)
# 3. Output:
#    - (e.g., Results of XYZ operation)
# 4. Prerequisites:
#    - (e.g., Necessary dependencies for XYZ)
# 
# (Remove me: Please fill docstrings above, while keeping the bulette point style, and remove this instruction line)
# """
# 
# from ._converters import (
#     _conversion_warning,
#     _return_if,
#     is_torch,
#     to_numpy,
#     to_torch,
# )
# from functools import wraps
# 
# from typing import Any as _Any
# from typing import Callable
# 
# import numpy as np
# import pandas as pd
# import torch
# 
# 
# def pandas_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         device = "cuda" if is_cuda(*args, **kwargs) else "cpu"
# 
#         def to_pandas(data: _Any) -> pd.DataFrame:
#             if isinstance(data, pd.DataFrame):
#                 return data
#             elif isinstance(data, pd.Series):
#                 return pd.DataFrame(data)
#             elif isinstance(data, (np.ndarray, list)):
#                 return pd.DataFrame(data)
#             elif isinstance(data, torch.Tensor):
#                 return pd.DataFrame(data.detach().cpu().numpy())
#             else:
#                 return pd.DataFrame([data])
# 
#         converted_args = [to_pandas(arg) for arg in args]
#         converted_kwargs = {key: to_pandas(val) for key, val in kwargs.items()}
#         results = func(*converted_args, **converted_kwargs)
#         if is_torch_input:
#             return to_torch(results, return_fn=_return_if, device=device)[0]
#         elif isinstance(results, (pd.DataFrame, pd.Series)):
#             return results
#         else:
#             return to_numpy(results, return_fn=_return_if)[0]
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

from ...src.mngs..decorators._pandas_fn import *

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
