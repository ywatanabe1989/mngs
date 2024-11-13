#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-14 07:55:43 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_modules/_matlab.py

from typing import Any

from pymatreader import read_mat


def _load_matlab(lpath: str, **kwargs) -> Any:
    """Load MATLAB file."""
    if not lpath.endswith(".mat"):
        raise ValueError("File must have .mat extension")
    return read_mat(lpath, **kwargs)


# EOF
