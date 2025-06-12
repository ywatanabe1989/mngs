#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-12 13:45:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/_yaml.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_modules/_yaml.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
YAML saving functionality for mngs.io.save
"""

from ruamel.yaml import YAML


def save_yaml(obj, spath, **kwargs):
    """Handle YAML file saving.
    
    Parameters
    ----------
    obj : dict, list, or YAML-serializable object
        Object to save as YAML file
    spath : str
        Path where YAML file will be saved
    **kwargs
        Additional keyword arguments passed to YAML().dump()
        
    Notes
    -----
    Uses ruamel.yaml for better preservation of formatting and comments
    """
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 4096  # Prevent unwanted line wrapping
    
    with open(spath, "w") as f:
        yaml.dump(obj, f, **kwargs)


# EOF