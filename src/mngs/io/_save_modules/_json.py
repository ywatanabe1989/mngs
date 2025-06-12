#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-12 13:45:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/_json.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_modules/_json.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
JSON saving functionality for mngs.io.save
"""

import json
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy arrays and dtypes."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_json(obj, spath, **kwargs):
    """Handle JSON file saving.
    
    Parameters
    ----------
    obj : dict, list, or JSON-serializable object
        Object to save as JSON file
    spath : str
        Path where JSON file will be saved
    **kwargs
        Additional keyword arguments passed to json.dump()
        
    Notes
    -----
    Uses custom encoder to handle NumPy arrays and dtypes
    """
    # Set default parameters for better readability
    if 'indent' not in kwargs:
        kwargs['indent'] = 2
    if 'cls' not in kwargs:
        kwargs['cls'] = NumpyJSONEncoder
        
    with open(spath, "w") as f:
        json.dump(obj, f, **kwargs)


# EOF
=======
# Timestamp: "2025-05-16 12:27:18 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_json.py

import json


def _save_json(obj, spath):
    """
    Save a Python object as a JSON file.
    
    Parameters
    ----------
    obj : dict or list
        The object to serialize to JSON.
    spath : str
        Path where the JSON file will be saved.
        
    Returns
    -------
    None
    """
    with open(spath, "w") as f:
        json.dump(obj, f, indent=4)
>>>>>>> origin/main
