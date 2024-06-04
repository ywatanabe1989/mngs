#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-04 18:16:32 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/general/_dict_replace.py


def dict_replace(string, dict):
    for k, v in dict.items():
        string = string.replace(k, v)
    return string
