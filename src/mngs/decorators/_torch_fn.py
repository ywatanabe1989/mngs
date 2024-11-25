#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-25 13:10:50 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/_torch_fn.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/_torch_fn.py"

"""
Functionality:
    - Implements PyTorch-specific conversion and utility functions
    - Provides decorators for PyTorch operations
Input:
    - Various data types to be converted to PyTorch tensors
Output:
    - PyTorch tensors and processing results
Prerequisites:
    - PyTorch package
    - Core converter utilities
"""

from functools import wraps
from typing import Any as _Any
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch

from ._converters import (
    _return_always,
    _return_if,
    is_torch,
    to_numpy,
    to_torch,
)



def torch_fn(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
        is_torch_input = is_torch(*args, **kwargs)

        # Convert inputs to torch tensors
        converted_args = to_torch(*args, **kwargs)
        if isinstance(converted_args, tuple):
            results = func(*converted_args)
        else:
            results = func(converted_args)

        # Convert output data based on input type
        if not is_torch_input:
            if isinstance(results, (list, tuple)):
                results = tuple(to_numpy(r) for r in results)
                if len(results) == 1:
                    results = results[0]
            else:
                results = to_numpy(results)

            # Only convert to list/Series if results is not a tuple
            if not isinstance(results, tuple):
                if isinstance(args[0], list):
                    results = results.tolist()
                elif isinstance(args[0], pd.Series):
                    results = pd.Series(results)
        return results

    return wrapper


def torch_method(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args: _Any, **kwargs: _Any) -> _Any:
        is_torch_input = is_torch(*args, **kwargs)
        if is_torch_input:
            results = func(self, *args, **kwargs)
        else:
            converted = to_torch(*args, return_fn=_return_always, **kwargs)
            converted_args = [
                arg[0] if isinstance(arg, tuple) else arg
                for arg in converted[0]
            ]
            converted_kwargs = {
                k: v[0] if isinstance(v, tuple) else v
                for k, v in converted[1].items()
            }
            results = func(self, *converted_args, **converted_kwargs)
            results = to_numpy(results, return_fn=_return_if)[0]
        return results

    return wrapper


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import torch.nn.functional as F

    def run_tests():
        @torch_fn
        def basic_operation(x: torch.Tensor) -> torch.Tensor:
            return x + 1.0

        @torch_fn
        def softmax_operation(x: torch.Tensor) -> torch.Tensor:
            return F.softmax(x, dim=-1)

        @torch_fn
        def multi_return(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return x * 2.0, x + 1.0

        # Test data
        test_inputs = [
            [1.0, 2.0, 3.0],
            np.array([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0]),
            pd.Series([1.0, 2.0, 3.0]),
        ]

        print("\nTest 1: Basic functionality")
        for data in test_inputs:
            result = basic_operation(data)
            print(f"Input type: {type(data)}, Output type: {type(result)}")

        print("\nTest 2: Softmax operation")
        for data in test_inputs:
            result = softmax_operation(data)
            print(f"Input type: {type(data)}, Output type: {type(result)}")

        print("\nTest 3: Multiple returns")
        result1, result2 = multi_return(np.array([1.0, 2.0, 3.0]))
        print(f"Output types: {type(result1)}, {type(result2)}")


if __name__ == "__main__":
    run_tests()
    import scipy
    import torch.nn.functional as F
    import numpy as np
    import mngs

    eeg_1min = np.random.rand(1, 16, 24000)
    pac, freqs_pha, freqs_amp = mngs.dsp.pac(
        eeg_1min,
        400,
        batch_size=1,
        batch_size_ch=8,
        fp16=True,
        n_perm=16,
    )

# EOF
