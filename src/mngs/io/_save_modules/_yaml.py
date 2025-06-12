#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
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
=======
# Timestamp: "2025-05-16 12:26:16 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_yaml.py
>>>>>>> origin/main

from ruamel.yaml import YAML


<<<<<<< HEAD
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
=======
def _save_yaml(obj, spath):
    """
    Save a Python object as a YAML file.
    
    Parameters
    ----------
    obj : dict
        The object to serialize to YAML.
    spath : str
        Path where the YAML file will be saved.
        
    Returns
    -------
    None
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=4, sequence=4, offset=4)

    with open(spath, "w") as f:
        yaml.dump(obj, f)
>>>>>>> origin/main
