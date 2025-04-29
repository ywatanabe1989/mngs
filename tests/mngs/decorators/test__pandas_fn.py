#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 15:45:06 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/decorators/test__pandas_fn.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/decorators/test__pandas_fn.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Mocking the is_cuda function as it's imported but not defined in the file
# This would normally be imported from converters
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from mngs.decorators._pandas_fn import pandas_fn


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


@patch("mngs.decorators._pandas_fn.is_cuda", return_value=False)
def test_pandas_fn_with_list_input(mock_is_cuda, test_data):
    """Test pandas_fn with list input."""

    @pandas_fn
    def dummy_function(df):
        # Check that input is indeed a pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        return df + 1.0

    # Input is a list, output should be numpy
    result = dummy_function(test_data["list"])
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array([2.0, 3.0, 4.0]))


@patch("mngs.decorators._pandas_fn.is_cuda", return_value=False)
def test_pandas_fn_with_numpy_input(mock_is_cuda, test_data):
    """Test pandas_fn with numpy input."""

    @pandas_fn
    def dummy_function(df):
        assert isinstance(df, pd.DataFrame)
        return df * 2.0

    # Input is numpy, output should be numpy
    result = dummy_function(test_data["numpy"])
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array([2.0, 4.0, 6.0]))


@patch("mngs.decorators._pandas_fn.is_cuda", return_value=False)
def test_pandas_fn_with_pandas_df_input(mock_is_cuda, test_data):
    """Test pandas_fn with pandas DataFrame input."""

    @pandas_fn
    def dummy_function(df):
        assert isinstance(df, pd.DataFrame)
        return df * 3.0

    # Input is pandas DF, output should remain pandas DF
    result = dummy_function(test_data["pandas_df"])
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(
        result, pd.DataFrame({"col1": [3.0, 6.0, 9.0]})
    )


@patch("mngs.decorators._pandas_fn.is_cuda", return_value=False)
def test_pandas_fn_with_pandas_series_input(mock_is_cuda, test_data):
    """Test pandas_fn with pandas Series input."""

    @pandas_fn
    def dummy_function(df):
        assert isinstance(df, pd.DataFrame)
        return df * 2.0

    # Input is pandas Series, output should be pandas DataFrame
    result = dummy_function(test_data["pandas_series"])
    assert isinstance(result, pd.DataFrame)
    expected = pd.DataFrame({"0": [2.0, 4.0, 6.0]})
    pd.testing.assert_frame_equal(result, expected)


@patch("mngs.decorators._pandas_fn.is_cuda", return_value=False)
@patch("mngs.decorators._pandas_fn.is_torch", return_value=True)
def test_pandas_fn_with_torch_input(mock_is_torch, mock_is_cuda, test_data):
    """Test pandas_fn with torch tensor input when is_torch returns True."""

    @pandas_fn
    def dummy_function(df):
        assert isinstance(df, pd.DataFrame)
        return df + 5.0

    # Mock to_torch to return a tensor when is_torch=True
    with patch(
        "mngs.decorators._pandas_fn.to_torch",
        return_value=(torch.tensor([6.0, 7.0, 8.0]),),
    ):
        result = dummy_function(test_data["torch"])
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, torch.tensor([6.0, 7.0, 8.0]))


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-26 18:46:08 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py"
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:55:46 (ywatanabe)"
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

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-26 18:46:08 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py"
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:55:46 (ywatanabe)"
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

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-26 18:46:08 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py"
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:55:46 (ywatanabe)"
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

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-26 18:46:08 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py"
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:55:46 (ywatanabe)"
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

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-26 18:46:08 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py"
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:55:46 (ywatanabe)"
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

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-26 18:46:08 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py"
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:55:46 (ywatanabe)"
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

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-26 18:46:08 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py"
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:55:46 (ywatanabe)"
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

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

# Mocking the is_cuda function as it's imported but not defined in the file
# This would normally be imported from converters


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


@patch("mngs.decorators._pandas_fn.is_cuda", return_value=False)
def test_pandas_fn_with_list_input(mock_is_cuda, test_data):
    """Test pandas_fn with list input."""

    @pandas_fn
    def dummy_function(df):
        # Check that input is indeed a pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        return df + 1.0

    # Input is a list, output should be numpy
    result = dummy_function(test_data["list"])
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array([2.0, 3.0, 4.0]))


@patch("mngs.decorators._pandas_fn.is_cuda", return_value=False)
def test_pandas_fn_with_numpy_input(mock_is_cuda, test_data):
    """Test pandas_fn with numpy input."""

    @pandas_fn
    def dummy_function(df):
        assert isinstance(df, pd.DataFrame)
        return df * 2.0

    # Input is numpy, output should be numpy
    result = dummy_function(test_data["numpy"])
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array([2.0, 4.0, 6.0]))


@patch("mngs.decorators._pandas_fn.is_cuda", return_value=False)
def test_pandas_fn_with_pandas_df_input(mock_is_cuda, test_data):
    """Test pandas_fn with pandas DataFrame input."""

    @pandas_fn
    def dummy_function(df):
        assert isinstance(df, pd.DataFrame)
        return df * 3.0

    # Input is pandas DF, output should remain pandas DF
    result = dummy_function(test_data["pandas_df"])
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(
        result, pd.DataFrame({"col1": [3.0, 6.0, 9.0]})
    )


@patch("mngs.decorators._pandas_fn.is_cuda", return_value=False)
def test_pandas_fn_with_pandas_series_input(mock_is_cuda, test_data):
    """Test pandas_fn with pandas Series input."""

    @pandas_fn
    def dummy_function(df):
        assert isinstance(df, pd.DataFrame)
        return df * 2.0

    # Input is pandas Series, output should be pandas DataFrame
    result = dummy_function(test_data["pandas_series"])
    assert isinstance(result, pd.DataFrame)
    expected = pd.DataFrame({"0": [2.0, 4.0, 6.0]})
    pd.testing.assert_frame_equal(result, expected)


@patch("mngs.decorators._pandas_fn.is_cuda", return_value=False)
@patch("mngs.decorators._pandas_fn.is_torch", return_value=True)
def test_pandas_fn_with_torch_input(mock_is_torch, mock_is_cuda, test_data):
    """Test pandas_fn with torch tensor input when is_torch returns True."""

    @pandas_fn
    def dummy_function(df):
        assert isinstance(df, pd.DataFrame)
        return df + 5.0

    # Mock to_torch to return a tensor when is_torch=True
    with patch(
        "mngs.decorators._pandas_fn.to_torch",
        return_value=(torch.tensor([6.0, 7.0, 8.0]),),
    ):
        result = dummy_function(test_data["torch"])
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, torch.tensor([6.0, 7.0, 8.0]))


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-26 18:46:08 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py"
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:55:46 (ywatanabe)"
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

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-26 18:46:08 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py"
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:55:46 (ywatanabe)"
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

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-26 18:46:08 (ywatanabe)"
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py"
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:55:46 (ywatanabe)"
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

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/decorators/_pandas_fn.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-26 18:46:08 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_pandas_fn.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py"
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

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/decorators/_pandas_fn.py
# --------------------------------------------------------------------------------
