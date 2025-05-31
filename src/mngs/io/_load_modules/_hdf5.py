#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-14 07:55:37 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_modules/_hdf5.py

from typing import Any

import h5py
import numpy as np


def _load_hdf5(lpath: str, **kwargs) -> Any:
    """Load HDF5 file."""
    if not (lpath.endswith(".hdf5") or lpath.endswith(".h5")):
        raise ValueError("File must have .hdf5 or .h5 extension")
    
    def load_item(item):
        """Recursively load items from HDF5."""
        if isinstance(item, h5py.Group):
            # Load groups as nested dicts
            result = {}
            for key in item.keys():
                result[key] = load_item(item[key])
            return result
        elif isinstance(item, h5py.Dataset):
            # Load datasets
            data = item[()]
            # Check if it's a pickled object
            if isinstance(data, np.void):
                import pickle
                return pickle.loads(data.tobytes())
            # Convert bytes to string
            elif isinstance(data, bytes):
                return data.decode('utf-8')
            # Convert numpy scalars to Python types
            elif isinstance(data, np.integer):
                return int(data)
            elif isinstance(data, np.floating):
                return float(data)
            elif isinstance(data, np.bool_):
                return bool(data)
            else:
                return data
        else:
            return item
    
    with h5py.File(lpath, "r") as hf:
        return load_item(hf)


# EOF
