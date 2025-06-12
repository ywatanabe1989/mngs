#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-12 13:45:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/_joblib.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_modules/_joblib.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Joblib saving functionality for mngs.io.save
"""
=======
# Timestamp: "2025-05-16 12:22:56 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_joblib.py
>>>>>>> origin/main

import joblib


<<<<<<< HEAD
def save_joblib(obj, spath, **kwargs):
    """Handle joblib file saving.
=======
def _save_joblib(obj, spath):
    """
    Save an object using joblib serialization.
>>>>>>> origin/main
    
    Parameters
    ----------
    obj : Any
<<<<<<< HEAD
        Object to save as joblib file
    spath : str
        Path where joblib file will be saved
    **kwargs
        Additional keyword arguments passed to joblib.dump()
    """
    joblib.dump(obj, spath, **kwargs)


# EOF
=======
        Object to serialize.
    spath : str
        Path where the joblib file will be saved.
        
    Returns
    -------
    None
    """
    with open(spath, "wb") as s:
        joblib.dump(obj, s, compress=3)
>>>>>>> origin/main
