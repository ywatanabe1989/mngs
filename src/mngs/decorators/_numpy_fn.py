#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 15:29:53 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_numpy_fn.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/decorators/_numpy_fn.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pandas as pd
import torch

THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_numpy_fn.py"

from functools import wraps
from typing import Any as _Any
from typing import Callable

from ._converters import _return_always, is_nested_decorator, to_numpy


def numpy_fn(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
        # Skip conversion if already in a nested decorator context
        if is_nested_decorator():
            results = func(*args, **kwargs)
            return results

        # Set the current decorator context
        wrapper._current_decorator = "numpy_fn"

        # Store original object for type preservation
        original_object = args[0] if args else None

        converted_args, converted_kwargs = to_numpy(
            *args, return_fn=_return_always, **kwargs
        )

        # Assertion to ensure all args are converted to numpy arrays
        for arg_index, arg in enumerate(converted_args):
            assert isinstance(
                arg, np.ndarray
            ), f"Argument {arg_index} not converted to numpy array: {type(arg)}"

        results = func(*converted_args, **converted_kwargs)

        # Convert results back to original input types
        if isinstance(results, np.ndarray):
            if original_object is not None:
                if isinstance(original_object, list):
                    return results.tolist()
                elif isinstance(original_object, torch.Tensor):
                    return torch.tensor(results)
                elif isinstance(original_object, pd.DataFrame):
                    return pd.DataFrame(results)
                elif isinstance(original_object, pd.Series):
                    return pd.Series(results)
            return results

        return results

    # Mark as a wrapper for detection
    wrapper._is_wrapper = True
    wrapper._decorator_type = "numpy_fn"
    return wrapper

# EOF