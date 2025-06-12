#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import joblib


def save_joblib(obj, spath, **kwargs):
    """Handle joblib file saving.
    
    Parameters
    ----------
    obj : Any
        Object to save as joblib file
    spath : str
        Path where joblib file will be saved
    **kwargs
        Additional keyword arguments passed to joblib.dump()
    """
    joblib.dump(obj, spath, **kwargs)


# EOF