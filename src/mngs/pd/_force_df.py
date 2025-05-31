#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 19:59:11 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/pd/_force_df.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/pd/_force_df.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pandas as pd

from ..types import is_listed_X


def force_df(permutable_dict, filler=np.nan):
    # Handle different input types
    if isinstance(permutable_dict, pd.DataFrame):
        return permutable_dict
    elif isinstance(permutable_dict, pd.Series):
        return permutable_dict.to_frame()
    elif is_listed_X(permutable_dict, pd.Series):
        permutable_dict = [sr.to_dict() for sr in permutable_dict]
    elif isinstance(permutable_dict, (list, tuple)):
        # Convert list/tuple to dict with default column name
        permutable_dict = {'0': permutable_dict}
    elif isinstance(permutable_dict, np.ndarray):
        if permutable_dict.ndim == 1:
            permutable_dict = {'0': permutable_dict}
        else:
            # For 2D arrays, create column names
            permutable_dict = {str(i): permutable_dict[:, i] for i in range(permutable_dict.shape[1])}
    
    ## Deep copy
    permutable_dict = permutable_dict.copy()

    ## Get the lengths
    max_len = 0
    for k, v in permutable_dict.items():
        # Check if v is an iterable (but not string) or treat as single length otherwise
        if isinstance(v, (str, int, float)) or not hasattr(v, "__len__"):
            length = 1
        else:
            length = len(v)
        max_len = max(max_len, length)

    ## Replace with appropriately filled list
    for k, v in permutable_dict.items():
        if isinstance(v, (str, int, float)) or not hasattr(v, "__len__"):
            permutable_dict[k] = [v] + [filler] * (max_len - 1)
        else:
            permutable_dict[k] = list(v) + [filler] * (max_len - len(v))

    ## Puts them into a DataFrame
    out_df = pd.DataFrame(permutable_dict)

    return out_df

# EOF