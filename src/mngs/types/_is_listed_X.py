#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 09:09:59 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/types/_is_listed_X.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/types/_is_listed_X.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

def is_listed_X(obj, types):
    """
    Example:
        obj = [3, 2, 1, 5]
        _is_listed_X(obj,
    """
    import numpy as np

    try:
        condition_list = isinstance(obj, list)

        if not (isinstance(types, list) or isinstance(types, tuple)):
            types = [types]

        _conditions_susp = []
        for typ in types:
            _conditions_susp.append(
                (np.array([isinstance(o, typ) for o in obj]) == True).all()
            )

        condition_susp = np.any(_conditions_susp)

        _is_listed_X = np.all([condition_list, condition_susp])
        return _is_listed_X

    except:
        return False

# EOF