#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 12:24:04 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_hdf5.py

import h5py


def _save_hdf5(obj, spath):
    """
    Save a dictionary of arrays to an HDF5 file.
    
    Parameters
    ----------
    obj : dict
        Dictionary of arrays to save. Keys will be dataset names.
    spath : str
        Path where the HDF5 file will be saved.
        
    Returns
    -------
    None
    """
    name_list, obj_list = [], []
    for k, v in obj.items():
        name_list.append(k)
        obj_list.append(v)
    with h5py.File(spath, "w") as hf:
        for name, obj in zip(name_list, obj_list):
            hf.create_dataset(name, data=obj)