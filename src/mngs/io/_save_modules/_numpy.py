#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-12 13:00:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/_numpy.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_modules/_numpy.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
NumPy saving functionality for mngs.io.save
"""
=======
# Timestamp: "2025-05-16 12:19:07 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_numpy.py
>>>>>>> origin/main

import numpy as np


<<<<<<< HEAD
def save_npy(obj, spath, **kwargs):
    """Handle NumPy .npy file saving.
    
    Parameters
    ----------
    obj : array-like
        Object to save as .npy file
    spath : str
        Path where .npy file will be saved
    **kwargs
        Additional keyword arguments (not used for .npy)
=======
def _save_npy(obj, spath):
    """
    Save a numpy array to .npy format.
    
    Parameters
    ----------
    obj : numpy.ndarray
        The numpy array to save.
    spath : str
        Path where the .npy file will be saved.
        
    Returns
    -------
    None
>>>>>>> origin/main
    """
    np.save(spath, obj)


<<<<<<< HEAD
def save_npz(obj, spath, **kwargs):
    """Handle NumPy .npz compressed file saving.
    
    Parameters
    ----------
    obj : dict, list, tuple, or np.ndarray
        Object to save as .npz file
        - dict: keys become array names in the npz file
        - list/tuple: arrays are saved with names "0", "1", etc.
        - single array: saved with name "arr_0"
    spath : str
        Path where .npz file will be saved
    **kwargs
        Additional keyword arguments passed to np.savez_compressed()
        
    Raises
    ------
    ValueError
        If object type is not supported for .npz format
    """
    if isinstance(obj, dict):
        np.savez_compressed(spath, **obj)
    elif isinstance(obj, (list, tuple)) and all(isinstance(x, np.ndarray) for x in obj):
        obj = {str(ii): obj[ii] for ii in range(len(obj))}
        np.savez_compressed(spath, **obj)
    elif isinstance(obj, np.ndarray):
        np.savez_compressed(spath, arr_0=obj)
    else:
        raise ValueError("For .npz files, obj must be a dict of arrays, a list/tuple of arrays, or a single array.")


# EOF
=======
def _save_npz(obj, spath):
    """
    Save numpy arrays to .npz format.
    
    Parameters
    ----------
    obj : dict or list/tuple of numpy.ndarray
        Either a dictionary of arrays or a list/tuple of arrays.
    spath : str
        Path where the .npz file will be saved.
        
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If obj is not a dict of arrays or a list/tuple of arrays.
    """
    if isinstance(obj, dict):
        np.savez_compressed(spath, **obj)
    elif isinstance(obj, (list, tuple)) and all(
        isinstance(x, np.ndarray) for x in obj
    ):
        obj = {str(ii): obj[ii] for ii in range(len(obj))}
        np.savez_compressed(spath, **obj)
    else:
        raise ValueError(
            "For .npz files, obj must be a dict of arrays or a list/tuple of arrays."
        )
>>>>>>> origin/main
