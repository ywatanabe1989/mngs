#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-14 07:41:24 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_modules/_hdf5.py

import h5py


def _load_hdf5(lpath, **kwargs):
    """Load HDF5 file."""
    if not lpath.endswith(".hdf5"):
        raise ValueError("File must have .hdf5 extension")
    obj = {}
    with h5py.File(lpath, "r") as hf:
        for name in hf:
            obj[name] = hf[name][:]
    return obj


# EOF
