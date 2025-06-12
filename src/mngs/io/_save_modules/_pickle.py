#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-12 13:45:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/_pickle.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_modules/_pickle.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Pickle saving functionality for mngs.io.save
"""
=======
# Timestamp: "2025-05-16 12:21:07 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_pickle.py
>>>>>>> origin/main

import pickle
import gzip


<<<<<<< HEAD
def save_pickle(obj, spath, **kwargs):
    """Handle pickle file saving.
    
    Parameters
    ----------
    obj : Any
        Object to save as pickle file
    spath : str
        Path where pickle file will be saved
    **kwargs
        Additional keyword arguments passed to pickle.dump()
    """
    with open(spath, "wb") as pkl_file:
        pickle.dump(obj, pkl_file, **kwargs)


def save_pickle_compressed(obj, spath, **kwargs):
    """Handle compressed pickle file saving (.pkl.gz).
=======
def _save_pickle(obj, spath):
    """
    Save an object using Python's pickle serialization.
>>>>>>> origin/main
    
    Parameters
    ----------
    obj : Any
<<<<<<< HEAD
        Object to save as compressed pickle file
    spath : str
        Path where compressed pickle file will be saved
    **kwargs
        Additional keyword arguments passed to pickle.dump()
    """
    with gzip.open(spath, "wb") as f:
        pickle.dump(obj, f, **kwargs)


# EOF
=======
        Object to serialize.
    spath : str
        Path where the pickle file will be saved.
        
    Returns
    -------
    None
    """
    with open(spath, "wb") as s:
        pickle.dump(obj, s)


def _save_pickle_gz(obj, spath):
    """
    Save an object using Python's pickle serialization with gzip compression.
    
    Parameters
    ----------
    obj : Any
        Object to serialize.
    spath : str
        Path where the compressed pickle file will be saved.
        
    Returns
    -------
    None
    """
    with gzip.open(spath, "wb") as f:
        pickle.dump(obj, f)
>>>>>>> origin/main
