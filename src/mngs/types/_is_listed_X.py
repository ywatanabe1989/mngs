#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 12:37:58 (ywatanabe)"
# File: ./mngs_repo/src/mngs/utils/_is_listed_X.py

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
