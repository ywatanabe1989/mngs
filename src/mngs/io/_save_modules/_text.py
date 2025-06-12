#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-12 13:45:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/_text.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_modules/_text.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Text file saving functionality for mngs.io.save

This module acts as a wrapper around the existing _save_text.py functionality
to maintain consistency with the modular save system.
"""

from ._save_text import _save_text


def save_text(obj, spath, **kwargs):
    """Handle text file saving.
    
    Parameters
    ----------
    obj : str
        Text content to save
    spath : str
        Path where text file will be saved
    **kwargs
        Additional keyword arguments (currently unused)
        
    Notes
    -----
    This is a wrapper around the existing _save_text function
    """
    _save_text(obj, spath)


# EOF