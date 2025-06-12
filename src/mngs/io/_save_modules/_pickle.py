#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import pickle
import gzip


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
    
    Parameters
    ----------
    obj : Any
        Object to save as compressed pickle file
    spath : str
        Path where compressed pickle file will be saved
    **kwargs
        Additional keyword arguments passed to pickle.dump()
    """
    with gzip.open(spath, "wb") as f:
        pickle.dump(obj, f, **kwargs)


# EOF