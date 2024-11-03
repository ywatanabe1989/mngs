#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 02:45:09 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/_numpy_fn.py

"""
1. Functionality:
   - (e.g., Executes XYZ operation)
2. Input:
   - (e.g., Required data for XYZ)
3. Output:
   - (e.g., Results of XYZ operation)
4. Prerequisites:
   - (e.g., Necessary dependencies for XYZ)

(Remove me: Please fill docstrings above, while keeping the bulette point style, and remove this instruction line)
"""

from functools import wraps
from typing import Any as _Any
from typing import Callable

from ._converters import (
    _conversion_warning,
    _return_always,
    _return_if,
    _try_device,
    is_cuda,
    is_torch,
    to_numpy,
    to_torch,
)


def numpy_fn(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
        is_torch_input = is_torch(*args, **kwargs)
        device = "cuda" if is_cuda(*args, **kwargs) else "cpu"
        converted_args, converted_kwargs = to_numpy(
            *args, return_fn=_return_always, **kwargs
        )
        results = func(*converted_args, **converted_kwargs)
        return (
            results
            if not is_torch_input
            else to_torch(results, return_fn=_return_if, device=device)[0][0]
        )

    return wrapper


# EOF
