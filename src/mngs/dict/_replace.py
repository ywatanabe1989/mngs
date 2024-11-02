#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "ywatanabe (2024-11-03 00:48:34)"
# File: ./mngs_repo/src/mngs/dict/_replace.py


def replace(string, dict):
    for k, v in dict.items():
        string = string.replace(k, v)
    return string


# EOF
