#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-12 13:45:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/_catboost.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_modules/_catboost.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
CatBoost model saving functionality for mngs.io.save
"""


def save_catboost(obj, spath, **kwargs):
    """Handle CatBoost model file saving.
    
    Parameters
    ----------
    obj : CatBoostClassifier, CatBoostRegressor, or CatBoost model
        CatBoost model object to save
    spath : str
        Path where CatBoost model file will be saved
    **kwargs
        Additional keyword arguments passed to model.save_model()
        
    Notes
    -----
    - The model must have a save_model method
    - Supports both binary (.cbm) and JSON formats
    """
    obj.save_model(spath, **kwargs)


# EOF