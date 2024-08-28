#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-26 10:06:11 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/general/_alternate_kwarg.py


def alternate_kwarg(kwargs, primary_key, alternate_key):
    alternate_value = kwargs.pop(alternate_key, None)
    kwargs[primary_key] = kwargs.get(primary_key) or alternate_value
    return kwargs
