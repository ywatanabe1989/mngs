#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 15:44:55 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/decorators/test__numpy_fn.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/decorators/test__numpy_fn.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Mocking the is_cuda function as it's imported but not defined in the file
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from mngs.decorators._numpy_fn import numpy_fn


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


@patch("mngs.decorators._numpy_fn.is_cuda", return_value=False)
def test_numpy_fn_with_list_input(mock_is_cuda, test_data):
    """Test numpy_fn with list input."""

    @numpy_fn
    def dummy_function(arr):
        # Check that input is indeed a numpy array
        assert isinstance(arr, np.ndarray)
        return arr + 1.0

    # Input is a list, output should be numpy
    with patch("mngs.decorators._numpy_fn.is_torch", return_value=False):
        with patch(
            "mngs.decorators._numpy_fn.to_numpy",
            return_value=([np.array([1.0, 2.0, 3.0])], {}),
        ):
            result = dummy_function(test_data["list"])
            assert isinstance(result, np.ndarray)
            np.testing.assert_allclose(result, np.array([2.0, 3.0, 4.0]))


@patch("mngs.decorators._numpy_fn.is_cuda", return_value=False)
def test_numpy_fn_with_numpy_input(mock_is_cuda, test_data):
    """Test numpy_fn with numpy input."""

    @numpy_fn
    def dummy_function(arr):
        assert isinstance(arr, np.ndarray)
        return arr * 2.0

    # Input is numpy, output should be numpy
    with patch("mngs.decorators._numpy_fn.is_torch", return_value=False):
        with patch(
            "mngs.decorators._numpy_fn.to_numpy",
            return_value=([np.array([1.0, 2.0, 3.0])], {}),
        ):
            result = dummy_function(test_data["numpy"])
            assert isinstance(result, np.ndarray)
            np.testing.assert_allclose(result, np.array([2.0, 4.0, 6.0]))


@patch("mngs.decorators._numpy_fn.is_cuda", return_value=False)
@patch("mngs.decorators._numpy_fn.is_torch", return_value=True)
def test_numpy_fn_with_torch_input(mock_is_torch, mock_is_cuda, test_data):
    """Test numpy_fn with torch tensor input when is_torch returns True."""

    @numpy_fn
    def dummy_function(arr):
        assert isinstance(arr, np.ndarray)
        return arr * 3.0

    # Mock to_numpy and to_torch to return appropriate values
    with patch(
        "mngs.decorators._numpy_fn.to_numpy",
        return_value=([np.array([1.0, 2.0, 3.0])], {}),
    ):
        with patch(
            "mngs.decorators._numpy_fn.to_torch",
            return_value=([[torch.tensor([3.0, 6.0, 9.0])]], {}),
        ):
            result = dummy_function(test_data["torch"])
            assert isinstance(result, torch.Tensor)
            torch.testing.assert_close(result, torch.tensor([3.0, 6.0, 9.0]))


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-01-18 03:44:07 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_numpy_fn.py"
#
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
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# import torch
# from ._converters import (
#     _conversion_warning,
#     _return_always,
#     _return_if,
#     _try_device,
#     is_cuda,
#     is_torch,
#     to_numpy,
#     to_torch,
# )
#
#
# def numpy_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         device = "cuda" if is_cuda(*args, **kwargs) else "cpu"
#         converted_args, converted_kwargs = to_numpy(
#             *args, return_fn=_return_always, **kwargs
#         )
#         results = func(*converted_args, **converted_kwargs)
#         return (
#             results
#             if not is_torch_input
#             else to_torch(results, return_fn=_return_if, device=device)[0][0]
#         )
#
#     return wrapper
#
#
# # EOF
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-01-18 03:44:07 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_numpy_fn.py"
#
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
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# import torch
# from ._converters import (
#     _conversion_warning,
#     _return_always,
#     _return_if,
#     _try_device,
#     is_cuda,
#     is_torch,
#     to_numpy,
#     to_torch,
# )
#
#
# def numpy_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         device = "cuda" if is_cuda(*args, **kwargs) else "cpu"
#         converted_args, converted_kwargs = to_numpy(
#             *args, return_fn=_return_always, **kwargs
#         )
#         results = func(*converted_args, **converted_kwargs)
#         return (
#             results
#             if not is_torch_input
#             else to_torch(results, return_fn=_return_if, device=device)[0][0]
#         )
#
#     return wrapper
#
#
# # EOF
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-01-18 03:44:07 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_numpy_fn.py"
#
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
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# import torch
# from ._converters import (
#     _conversion_warning,
#     _return_always,
#     _return_if,
#     _try_device,
#     is_cuda,
#     is_torch,
#     to_numpy,
#     to_torch,
# )
#
#
# def numpy_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         device = "cuda" if is_cuda(*args, **kwargs) else "cpu"
#         converted_args, converted_kwargs = to_numpy(
#             *args, return_fn=_return_always, **kwargs
#         )
#         results = func(*converted_args, **converted_kwargs)
#         return (
#             results
#             if not is_torch_input
#             else to_torch(results, return_fn=_return_if, device=device)[0][0]
#         )
#
#     return wrapper
#
#
# # EOF
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-01-18 03:44:07 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_numpy_fn.py"
#
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
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# import torch
# from ._converters import (
#     _conversion_warning,
#     _return_always,
#     _return_if,
#     _try_device,
#     is_cuda,
#     is_torch,
#     to_numpy,
#     to_torch,
# )
#
#
# def numpy_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         device = "cuda" if is_cuda(*args, **kwargs) else "cpu"
#         converted_args, converted_kwargs = to_numpy(
#             *args, return_fn=_return_always, **kwargs
#         )
#         results = func(*converted_args, **converted_kwargs)
#         return (
#             results
#             if not is_torch_input
#             else to_torch(results, return_fn=_return_if, device=device)[0][0]
#         )
#
#     return wrapper
#
#
# # EOF
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-01-18 03:44:07 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_numpy_fn.py"
#
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
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# import torch
# from ._converters import (
#     _conversion_warning,
#     _return_always,
#     _return_if,
#     _try_device,
#     is_cuda,
#     is_torch,
#     to_numpy,
#     to_torch,
# )
#
#
# def numpy_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         device = "cuda" if is_cuda(*args, **kwargs) else "cpu"
#         converted_args, converted_kwargs = to_numpy(
#             *args, return_fn=_return_always, **kwargs
#         )
#         results = func(*converted_args, **converted_kwargs)
#         return (
#             results
#             if not is_torch_input
#             else to_torch(results, return_fn=_return_if, device=device)[0][0]
#         )
#
#     return wrapper
#
#
# # EOF
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-01-18 03:44:07 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_numpy_fn.py"
#
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
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# import torch
# from ._converters import (
#     _conversion_warning,
#     _return_always,
#     _return_if,
#     _try_device,
#     is_cuda,
#     is_torch,
#     to_numpy,
#     to_torch,
# )
#
#
# def numpy_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         device = "cuda" if is_cuda(*args, **kwargs) else "cpu"
#         converted_args, converted_kwargs = to_numpy(
#             *args, return_fn=_return_always, **kwargs
#         )
#         results = func(*converted_args, **converted_kwargs)
#         return (
#             results
#             if not is_torch_input
#             else to_torch(results, return_fn=_return_if, device=device)[0][0]
#         )
#
#     return wrapper
#
#
# # EOF
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

# Mocking the is_cuda function as it's imported but not defined in the file


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


@patch("mngs.decorators._numpy_fn.is_cuda", return_value=False)
def test_numpy_fn_with_list_input(mock_is_cuda, test_data):
    """Test numpy_fn with list input."""

    @numpy_fn
    def dummy_function(arr):
        # Check that input is indeed a numpy array
        assert isinstance(arr, np.ndarray)
        return arr + 1.0

    # Input is a list, output should be numpy
    with patch("mngs.decorators._numpy_fn.is_torch", return_value=False):
        with patch(
            "mngs.decorators._numpy_fn.to_numpy",
            return_value=([np.array([1.0, 2.0, 3.0])], {}),
        ):
            result = dummy_function(test_data["list"])
            assert isinstance(result, np.ndarray)
            np.testing.assert_allclose(result, np.array([2.0, 3.0, 4.0]))


@patch("mngs.decorators._numpy_fn.is_cuda", return_value=False)
def test_numpy_fn_with_numpy_input(mock_is_cuda, test_data):
    """Test numpy_fn with numpy input."""

    @numpy_fn
    def dummy_function(arr):
        assert isinstance(arr, np.ndarray)
        return arr * 2.0

    # Input is numpy, output should be numpy
    with patch("mngs.decorators._numpy_fn.is_torch", return_value=False):
        with patch(
            "mngs.decorators._numpy_fn.to_numpy",
            return_value=([np.array([1.0, 2.0, 3.0])], {}),
        ):
            result = dummy_function(test_data["numpy"])
            assert isinstance(result, np.ndarray)
            np.testing.assert_allclose(result, np.array([2.0, 4.0, 6.0]))


@patch("mngs.decorators._numpy_fn.is_cuda", return_value=False)
@patch("mngs.decorators._numpy_fn.is_torch", return_value=True)
def test_numpy_fn_with_torch_input(mock_is_torch, mock_is_cuda, test_data):
    """Test numpy_fn with torch tensor input when is_torch returns True."""

    @numpy_fn
    def dummy_function(arr):
        assert isinstance(arr, np.ndarray)
        return arr * 3.0

    # Mock to_numpy and to_torch to return appropriate values
    with patch(
        "mngs.decorators._numpy_fn.to_numpy",
        return_value=([np.array([1.0, 2.0, 3.0])], {}),
    ):
        with patch(
            "mngs.decorators._numpy_fn.to_torch",
            return_value=([[torch.tensor([3.0, 6.0, 9.0])]], {}),
        ):
            result = dummy_function(test_data["torch"])
            assert isinstance(result, torch.Tensor)
            torch.testing.assert_close(result, torch.tensor([3.0, 6.0, 9.0]))


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-01-18 03:44:07 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_numpy_fn.py"
#
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
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# import torch
# from ._converters import (
#     _conversion_warning,
#     _return_always,
#     _return_if,
#     _try_device,
#     is_cuda,
#     is_torch,
#     to_numpy,
#     to_torch,
# )
#
#
# def numpy_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         device = "cuda" if is_cuda(*args, **kwargs) else "cpu"
#         converted_args, converted_kwargs = to_numpy(
#             *args, return_fn=_return_always, **kwargs
#         )
#         results = func(*converted_args, **converted_kwargs)
#         return (
#             results
#             if not is_torch_input
#             else to_torch(results, return_fn=_return_if, device=device)[0][0]
#         )
#
#     return wrapper
#
#
# # EOF
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-01-18 03:44:07 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_numpy_fn.py"
#
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
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# import torch
# from ._converters import (
#     _conversion_warning,
#     _return_always,
#     _return_if,
#     _try_device,
#     is_cuda,
#     is_torch,
#     to_numpy,
#     to_torch,
# )
#
#
# def numpy_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         device = "cuda" if is_cuda(*args, **kwargs) else "cpu"
#         converted_args, converted_kwargs = to_numpy(
#             *args, return_fn=_return_always, **kwargs
#         )
#         results = func(*converted_args, **converted_kwargs)
#         return (
#             results
#             if not is_torch_input
#             else to_torch(results, return_fn=_return_if, device=device)[0][0]
#         )
#
#     return wrapper
#
#
# # EOF
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-01-18 03:44:07 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_numpy_fn.py"
#
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
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# import torch
# from ._converters import (
#     _conversion_warning,
#     _return_always,
#     _return_if,
#     _try_device,
#     is_cuda,
#     is_torch,
#     to_numpy,
#     to_torch,
# )
#
#
# def numpy_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         device = "cuda" if is_cuda(*args, **kwargs) else "cpu"
#         converted_args, converted_kwargs = to_numpy(
#             *args, return_fn=_return_always, **kwargs
#         )
#         results = func(*converted_args, **converted_kwargs)
#         return (
#             results
#             if not is_torch_input
#             else to_torch(results, return_fn=_return_if, device=device)[0][0]
#         )
#
#     return wrapper
#
#
# # EOF
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/decorators/_numpy_fn.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-01-18 03:44:07 (ywatanabe)"
# # File: _numpy_fn.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_numpy_fn.py"
# 
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
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# import torch
# from ._converters import (
#     _conversion_warning,
#     _return_always,
#     _return_if,
#     _try_device,
#     is_cuda,
#     is_torch,
#     to_numpy,
#     to_torch,
# )
# 
# 
# def numpy_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         is_torch_input = is_torch(*args, **kwargs)
#         device = "cuda" if is_cuda(*args, **kwargs) else "cpu"
#         converted_args, converted_kwargs = to_numpy(
#             *args, return_fn=_return_always, **kwargs
#         )
#         results = func(*converted_args, **converted_kwargs)
#         return (
#             results
#             if not is_torch_input
#             else to_torch(results, return_fn=_return_if, device=device)[0][0]
#         )
# 
#     return wrapper
# 
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/decorators/_numpy_fn.py
# --------------------------------------------------------------------------------
