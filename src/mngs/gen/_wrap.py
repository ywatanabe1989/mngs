#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-06 07:39:27 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/gen/_wrap.py


# import functools


# def wrap(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         return func(*args, **kwargs)

#     return wrapper


import functools


def wrap(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
