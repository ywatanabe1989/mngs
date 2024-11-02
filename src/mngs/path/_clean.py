#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-25 12:18:55 (ywatanabe)"
# /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/str/_clean_path.py

def clean(string):
    string = string.replace("/./", "/")
    return string
