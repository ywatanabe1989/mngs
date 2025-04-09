#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-03-17 14:11:23 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/io/_load_modules/_matlab.py
# ----------------------------------------
import os
__FILE__ = (
    "/home/ywatanabe/proj/mngs_repo/src/mngs/io/_load_modules/_matlab.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from typing import Any
import h5py
import numpy as np

def _load_matlab(lpath: str, **kwargs) -> Any:
    """Load MATLAB file."""
    if not lpath.endswith(".mat"):
        raise ValueError("File must have .mat extension")

    try:
        # First try pymatreader for older MAT files
        from pymatreader import read_mat
        return read_mat(lpath, **kwargs)
    except Exception:
        # For MATLAB v7.3 files (HDF5 format)
        try:
            return _load_matlab_v73(lpath, **kwargs)
        except Exception as e:
            raise ValueError(f"Error loading MATLAB file: {str(e)}")

def _load_matlab_v73(lpath: str, **kwargs) -> Any:
    """Load MATLAB v7.3 files using h5py."""
    result = {}
    with h5py.File(lpath, 'r') as f:
        # Recursive function to extract all datasets
        def extract_datasets(group, path=""):
            for key, item in group.items():
                path_name = f"{path}/{key}" if path else key
                if isinstance(item, h5py.Dataset):
                    # Load dataset and convert to numpy array
                    result[path_name] = item[()]
                    # Convert byte strings to regular strings
                    if isinstance(result[path_name], np.ndarray) and result[path_name].dtype.kind == 'S':
                        result[path_name] = np.char.decode(result[path_name], 'utf-8')
                elif isinstance(item, h5py.Group):
                    # Recursively extract from this group
                    extract_datasets(item, path_name)

        extract_datasets(f)

    return result

# EOF