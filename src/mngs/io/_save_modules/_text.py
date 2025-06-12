#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
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
=======
# Timestamp: "2025-05-16 12:17:12 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_text.py


def _save_text(obj, spath):
    """
    Save text content to a file.
>>>>>>> origin/main
    
    Parameters
    ----------
    obj : str
<<<<<<< HEAD
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
=======
        The text content to save.
    spath : str
        Path where the text file will be saved.
        
    Returns
    -------
    None
    """
    with open(spath, "w") as file:
        file.write(obj)
>>>>>>> origin/main
