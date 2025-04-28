#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 15:45:43 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/decorators/test__torch_fn.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/decorators/test__torch_fn.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pandas as pd
import pytest
import torch
from mngs.decorators._torch_fn import torch_fn


@pytest.fixture
def test_data():
    """Create test data for tests."""
    return {
        "list": [1.0, 2.0, 3.0],
        "numpy": np.array([1.0, 2.0, 3.0]),
        "pandas_series": pd.Series([1.0, 2.0, 3.0]),
        "pandas_df": pd.DataFrame({"col1": [1.0, 2.0, 3.0]}),
        "torch": torch.tensor([1.0, 2.0, 3.0]),
    }


def test_torch_fn_with_list_input(test_data):
    """Test torch_fn with list input."""

    @torch_fn
    def dummy_function(xx):
        # Check that input is indeed a torch tensor
        assert isinstance(xx, torch.Tensor)
        return xx + 1.0

    # Input is a list, output should be numpy
    result = dummy_function(test_data["list"])
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array([2.0, 3.0, 4.0]))


def test_torch_fn_with_numpy_input(test_data):
    """Test torch_fn with numpy input."""

    @torch_fn
    def dummy_function(xx):
        assert isinstance(xx, torch.Tensor)
        return xx * 2.0

    # Input is numpy, output should be numpy
    result = dummy_function(test_data["numpy"])
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array([2.0, 4.0, 6.0]))


def test_torch_fn_with_torch_input(test_data):
    """Test torch_fn with torch input."""

    @torch_fn
    def dummy_function(xx):
        assert isinstance(xx, torch.Tensor)
        return xx * 3.0

    # Input is torch, output should remain torch
    result = dummy_function(test_data["torch"])
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, torch.tensor([3.0, 6.0, 9.0]))


def test_torch_fn_with_pandas_input(test_data):
    """Test torch_fn with pandas input."""

    @torch_fn
    def dummy_function(xx):
        assert isinstance(xx, torch.Tensor)
        return xx + 5.0

    # Test with pandas Series
    result_series = dummy_function(test_data["pandas_series"])
    assert isinstance(result_series, np.ndarray)
    np.testing.assert_allclose(result_series, np.array([6.0, 7.0, 8.0]))

    # Test with pandas DataFrame
    result_df = dummy_function(test_data["pandas_df"])
    assert isinstance(result_df, np.ndarray)
    np.testing.assert_allclose(result_df, np.array([6.0, 7.0, 8.0]))


def test_torch_fn_complex_operation(test_data):
    """Test torch_fn with a more complex operation."""

    @torch_fn
    def softmax_function(xx):
        assert isinstance(xx, torch.Tensor)
        return torch.nn.functional.softmax(xx, dim=0)

    # Test with numpy array
    result = softmax_function(test_data["numpy"])
    # Calculate expected softmax manually
    exp_values = np.exp(test_data["numpy"])
    expected = exp_values / np.sum(exp_values)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-24 15:38:15 (ywatanabe)"
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/decorators/_torch_fn.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-12-05 09:23:06 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_torch_fn.py"
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-06 15:45:12 (ywatanabe)"
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
# def torch_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         converted_args, converted_kwargs = to_torch(
#             *args, return_fn=_return_always, **kwargs
#         )
#         results = func(*converted_args, **converted_kwargs)
#         # print(type(results))
#         return (
#             to_numpy(results, return_fn=_return_if)[0]
#             if not is_torch_input
#             else results
#         )
#
#     return wrapper
#
# # def torch_fn(func: Callable) -> Callable:
# #     @wraps(func)
# #     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
# #         is_torch_input = is_torch(*args, **kwargs)
# #         # Skip conversion if inputs are already torch tensors
# #         if is_torch_input:
# #             results = func(*args, **kwargs)
# #         else:
# #             converted_args, converted_kwargs = to_torch(
# #                 *args, return_fn=_return_always, **kwargs
# #             )
# #             results = func(*converted_args, **converted_kwargs)
# #             results = to_numpy(results, return_fn=_return_if)[0]
# #         return results
#
# #     return wrapper
#
#

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/decorators/_torch_fn.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-24 15:38:15 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_torch_fn.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/decorators/_torch_fn.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-12-05 09:23:06 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_torch_fn.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_torch_fn.py"
# 
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
# def torch_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         converted_args, converted_kwargs = to_torch(
#             *args, return_fn=_return_always, **kwargs
#         )
#         results = func(*converted_args, **converted_kwargs)
#         # print(type(results))
#         return (
#             to_numpy(results, return_fn=_return_if)[0]
#             if not is_torch_input
#             else results
#         )
# 
#     return wrapper
# 
# # def torch_fn(func: Callable) -> Callable:
# #     @wraps(func)
# #     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
# #         is_torch_input = is_torch(*args, **kwargs)
# #         # Skip conversion if inputs are already torch tensors
# #         if is_torch_input:
# #             results = func(*args, **kwargs)
# #         else:
# #             converted_args, converted_kwargs = to_torch(
# #                 *args, return_fn=_return_always, **kwargs
# #             )
# #             results = func(*converted_args, **converted_kwargs)
# #             results = to_numpy(results, return_fn=_return_if)[0]
# #         return results
# 
# #     return wrapper
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
# """
# python ./mngs_repo/src/mngs/decorators/_torch_fn.py
# python -m src.mngs.decorators._torch_fn
# """
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/decorators/_torch_fn.py
# --------------------------------------------------------------------------------
