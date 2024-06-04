#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-04 08:32:27 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/general/_wrap.py


import functools


def wrap(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
