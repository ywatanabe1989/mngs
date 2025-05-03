#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 15:44:00 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/decorators/_pandas_fn.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_pandas_fn.py"

from functools import wraps
from typing import Any as _Any
from typing import Callable

import numpy as np
import pandas as pd
import torch
import xarray as xr

from ._converters import is_nested_decorator


def pandas_fn(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
        # Skip conversion if already in a nested decorator context
        if is_nested_decorator():
            results = func(*args, **kwargs)
            return results

        # Set the current decorator context
        wrapper._current_decorator = "pandas_fn"

        # Store original object for type preservation
        original_object = args[0] if args else None

        # Convert args to pandas DataFrames
        def to_pandas(data):
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, pd.Series):
                return pd.DataFrame(data)
            elif isinstance(data, np.ndarray):
                return pd.DataFrame(data)
            elif isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, torch.Tensor):
                return pd.DataFrame(data.detach().cpu().numpy())
            elif isinstance(data, xr.DataArray):
                return pd.DataFrame(data.values)
            else:
                return pd.DataFrame([data])

        converted_args = [to_pandas(arg) for arg in args]
        converted_kwargs = {k: to_pandas(v) for k, v in kwargs.items()}

        # Assertion to ensure all args are converted to pandas DataFrames
        for arg_index, arg in enumerate(converted_args):
            assert isinstance(
                arg, pd.DataFrame
            ), f"Argument {arg_index} not converted to DataFrame: {type(arg)}"

        results = func(*converted_args, **converted_kwargs)

        # Convert results back to original input types
        if isinstance(results, pd.DataFrame):
            if original_object is not None:
                if isinstance(original_object, list):
                    return results.values.tolist()
                elif isinstance(original_object, np.ndarray):
                    return results.values
                elif isinstance(original_object, torch.Tensor):
                    return torch.tensor(results.values)
                elif isinstance(original_object, pd.Series):
                    return (
                        pd.Series(results.iloc[:, 0])
                        if results.shape[1] > 0
                        else pd.Series()
                    )
                elif isinstance(original_object, xr.DataArray):
                    return xr.DataArray(results.values)
            return results

        return results

    # Mark as a wrapper for detection
    wrapper._is_wrapper = True
    wrapper._decorator_type = "pandas_fn"
    return wrapper

# EOF