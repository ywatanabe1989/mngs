#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 15:49:06 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/decorators/test__torch_fn.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/decorators/test__torch_fn.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
from functools import wraps

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
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
        "xarray": xr.DataArray([1.0, 2.0, 3.0]),
    }


def test_torch_fn_with_list_input(test_data):
    """Test torch_fn with list input."""

    @torch_fn
    def dummy_function(arr):
        # Check that input is indeed a torch tensor
        assert isinstance(arr, torch.Tensor)
        return arr + 1.0

    # Input is a list, output should be list
    with patch(
        "mngs.decorators._torch_fn.to_torch",
        return_value=([torch.tensor([1.0, 2.0, 3.0])], {}),
    ):
        result = dummy_function(test_data["list"])
        assert isinstance(result, list)
        assert result == [2.0, 3.0, 4.0]


def test_torch_fn_with_torch_input(test_data):
    """Test torch_fn with torch input."""

    @torch_fn
    def dummy_function(arr):
        assert isinstance(arr, torch.Tensor)
        return arr * 2.0

    # Input is torch, output should be torch
    with patch(
        "mngs.decorators._torch_fn.to_torch",
        return_value=([torch.tensor([1.0, 2.0, 3.0])], {}),
    ):
        result = dummy_function(test_data["torch"])
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, torch.tensor([2.0, 4.0, 6.0]))


def test_torch_fn_with_numpy_input(test_data):
    """Test torch_fn with numpy input."""

    @torch_fn
    def dummy_function(arr):
        assert isinstance(arr, torch.Tensor)
        return arr * 3.0

    # Input is numpy, output should be numpy
    with patch(
        "mngs.decorators._torch_fn.to_torch",
        return_value=([torch.tensor([1.0, 2.0, 3.0])], {}),
    ):
        result = dummy_function(test_data["numpy"])
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, np.array([3.0, 6.0, 9.0]))


def test_torch_fn_nested_decorator(test_data):
    """Test nested decorator behavior with torch_fn."""

    # Create a dummy decorator to simulate nesting
    def dummy_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set nested context
            wrapper._current_decorator = "dummy_decorator"
            return func(*args, **kwargs)

        wrapper._is_wrapper = True
        return wrapper

    # Apply both decorators (nested)
    @torch_fn
    @dummy_decorator
    def nested_function(arr):
        # In nested mode, the type should pass through unchanged from dummy_decorator
        assert not isinstance(arr, torch.Tensor)
        return arr

    with patch(
        "mngs.decorators._torch_fn.is_nested_decorator", return_value=True
    ):
        # Input numpy should stay as numpy due to nested context
        result = nested_function(test_data["numpy"])
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, test_data["numpy"])

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/decorators/_torch_fn.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-30 15:40:43 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_torch_fn.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/decorators/_torch_fn.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# 
# import numpy as np
# import pandas as pd
# import torch
# import xarray as xr
# 
# from ._converters import _return_always, is_nested_decorator, to_torch
# 
# 
# def torch_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         # Skip conversion if already in a nested decorator context
#         if is_nested_decorator():
#             results = func(*args, **kwargs)
#             return results
# 
#         # Set the current decorator context
#         wrapper._current_decorator = "torch_fn"
# 
#         # Store original object for type preservation
#         original_object = args[0] if args else None
# 
#         converted_args, converted_kwargs = to_torch(
#             *args, return_fn=_return_always, **kwargs
#         )
# 
#         # Assertion to ensure all args are converted to torch tensors
#         for arg_index, arg in enumerate(converted_args):
#             assert isinstance(
#                 arg, torch.Tensor
#             ), f"Argument {arg_index} not converted to torch.Tensor: {type(arg)}"
# 
#         results = func(*converted_args, **converted_kwargs)
# 
#         # Convert results back to original input types
#         if isinstance(results, torch.Tensor):
#             if original_object is not None:
#                 if isinstance(original_object, list):
#                     return results.detach().cpu().numpy().tolist()
#                 elif isinstance(original_object, np.ndarray):
#                     return results.detach().cpu().numpy()
#                 elif isinstance(original_object, pd.DataFrame):
#                     return pd.DataFrame(results.detach().cpu().numpy())
#                 elif isinstance(original_object, pd.Series):
#                     return pd.Series(results.detach().cpu().numpy().flatten())
#                 elif isinstance(original_object, xr.DataArray):
#                     return xr.DataArray(results.detach().cpu().numpy())
#             return results
# 
#         return results
# 
#     # Mark as a wrapper for detection
#     wrapper._is_wrapper = True
#     wrapper._decorator_type = "torch_fn"
#     return wrapper
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/decorators/_torch_fn.py
# --------------------------------------------------------------------------------
