#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-12 13:45:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/_image.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_modules/_image.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Image file saving functionality for mngs.io.save

This module acts as a wrapper around the existing _save_image.py functionality
to maintain consistency with the modular save system.
"""

from ._save_image import _save_image


def save_image(obj, spath, **kwargs):
    """Handle image file saving.
    
    Parameters
    ----------
    obj : matplotlib.figure.Figure, matplotlib.axes.Axes, PIL.Image, or array-like
        Image object to save. Can be a matplotlib figure/axes, PIL Image, or numpy array
    spath : str
        Path where image file will be saved
    **kwargs
        Additional keyword arguments passed to the underlying save function
        
    Notes
    -----
    This is a wrapper around the existing _save_image function
    Supports formats: PNG, JPG, JPEG, GIF, TIFF, TIF, SVG
    """
    _save_image(obj, spath, **kwargs)


# EOF