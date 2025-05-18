#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-03 06:25:32 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_save_text.py

def _save_text(obj, spath):
    with open(spath, "w") as file:
        file.write(obj)


# EOF
