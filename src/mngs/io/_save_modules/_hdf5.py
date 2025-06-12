#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-12 13:46:53 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/_hdf5.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/io/_save_modules/_hdf5.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np

"""
HDF5 saving functionality for mngs.io.save
"""
=======
# Timestamp: "2025-05-16 12:24:04 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_hdf5.py
>>>>>>> origin/main

import h5py


<<<<<<< HEAD
def save_hdf5(obj, spath, group_path=None, **kwargs):
    """Handle HDF5 file saving with automatic group/root switching.

    Parameters
    ----------
    obj : dict
        Dictionary where keys are dataset names and values are data arrays
    spath : str
        Path where HDF5 file will be saved
    group_path : str, optional
        If provided, save to specific group. If None, save to root.
    **kwargs
        Additional keyword arguments passed to h5py.File.create_dataset()

    Examples
    --------
    >>> # Save to root
    >>> data = {'array1': np.array([1, 2, 3])}
    >>> save_hdf5(data, 'data.hdf5')

    >>> # Save to group
    >>> save_hdf5(data, 'data.hdf5', group_path='experiment1/trial1')
    """
    with h5py.File(spath, "a") as hf:
        if group_path:
            if group_path in hf:
                del hf[group_path]
            target = hf.create_group(group_path)
        else:
            target = hf

        for key, value in obj.items():
            if isinstance(value, np.ndarray):
                target.create_dataset(
                    key, data=value, compression="gzip", **kwargs
                )
            else:
                target.attrs[key] = value


# def save_hdf5(obj, spath, **kwargs):
#     """Handle HDF5 file saving.

#     Parameters
#     ----------
#     obj : dict
#         Dictionary where keys are dataset names and values are data arrays
#     spath : str
#         Path where HDF5 file will be saved
#     **kwargs
#         Additional keyword arguments passed to h5py.File.create_dataset()

#     Notes
#     -----
#     - Expects obj to be a dictionary mapping dataset names to data
#     - Supports compression via kwargs (e.g., compression='gzip')

#     Examples
#     --------
#     >>> data = {'array1': np.array([1, 2, 3]), 'array2': np.array([[1, 2], [3, 4]])}
#     >>> save_hdf5(data, 'data.hdf5', compression='gzip')
#     """
#     with h5py.File(spath, "w") as hf:
#         for key, value in obj.items():
#             hf.create_dataset(key, data=value, **kwargs)


# def _save_hdf5_group(obj: dict, spath: str, group_path: str, **kwargs):
#     """Save to specific group in HDF5 file."""
#     with h5py.File(spath, "a") as hf:
#         if group_path in hf:
#             del hf[group_path]

#         grp = hf.create_group(group_path)
#         for key, value in obj.items():
#             if isinstance(value, np.ndarray):
#                 grp.create_dataset(key, data=value, compression="gzip")
#             else:
#                 grp.attrs[key] = value

# EOF
=======
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
>>>>>>> origin/main
