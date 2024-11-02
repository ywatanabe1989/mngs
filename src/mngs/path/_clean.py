#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 16:51:46 (ywatanabe)"
# File: ./mngs_repo/src/mngs/path/_clean.py

def clean(string):
    string = string.replace("/./", "/").replace("//", "/").replace(" ", "_")
    return string


# EOF
