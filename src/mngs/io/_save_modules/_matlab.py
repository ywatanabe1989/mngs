#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-12 13:45:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/_matlab.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_modules/_matlab.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
MATLAB .mat file saving functionality for mngs.io.save
"""

import scipy.io


def save_matlab(obj, spath, **kwargs):
    """Handle MATLAB .mat file saving.
    
    Parameters
    ----------
    obj : dict
        Dictionary where keys are variable names and values are data
    spath : str
        Path where .mat file will be saved
    **kwargs
        Additional keyword arguments passed to scipy.io.savemat()
        
    Notes
    -----
    - Expects obj to be a dictionary mapping variable names to data
    - Variable names must be valid MATLAB identifiers
    - Supports both v5 and v7.3 formats via kwargs
    
    Examples
    --------
    >>> data = {'x': np.array([1, 2, 3]), 'y': np.array([[1, 2], [3, 4]])}
    >>> save_matlab(data, 'data.mat')
    """
    scipy.io.savemat(spath, obj, **kwargs)


# EOF