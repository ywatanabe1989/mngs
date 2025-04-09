#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-17 20:47:30 (ywatanabe)"
# File: ./mngs_repo/src/mngs/path/_clean.py

THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/path/_clean.py"

def clean(string):
    string = string.replace("/./", "/").replace("//", "/").replace(" ", "_")
    return string


# EOF
