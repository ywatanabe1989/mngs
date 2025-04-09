#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-04 22:06:47 (ywatanabe)"
# File: /ssh:ywatanabe@spartan:/home/ywatanabe/proj/mngs_repo/src/mngs/io/_load_modules/_matlab.py
# ----------------------------------------
import os
__FILE__ = (
    "/ssh:ywatanabe@spartan:/home/ywatanabe/proj/mngs_repo/src/mngs/io/_load_modules/_matlab.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from typing import Any
import warnings

def _load_matlab(lpath: str, **kwargs) -> Any:
    """Load MATLAB file."""
    if not lpath.endswith(".mat"):
        raise ValueError("File must have .mat extension")

    # Try using scipy.io first for binary .mat files
    try:
        from scipy.io import loadmat
        return loadmat(lpath, **kwargs)
    except Exception as e1:
        # If scipy fails, try pymatreader
        try:
            from pymatreader import read_mat
            return read_mat(lpath, **kwargs)
        except Exception as e2:
            # Both methods failed
            raise ValueError(f"Error loading file {lpath}: {str(e1)}\nAnd: {str(e2)}")

# EOF