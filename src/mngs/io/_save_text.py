#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 17:00:45 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_save_text.py

def save_text(obj, spath):
    with open(spath, "w") as file:
        file.write(obj)


# EOF
