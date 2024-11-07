#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-07 05:57:34 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/_wrap.py

import functools


def wrap(func):
    """Basic function wrapper that preserves function metadata.

    Usage:
        @wrap
        def my_function(x):
            return x + 1

        # Or manually:
        def my_function(x):
            return x + 1
        wrapped_func = wrap(my_function)

    This wrapper is useful as a template for creating more complex decorators
    or when you want to ensure function metadata is preserved.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# EOF
