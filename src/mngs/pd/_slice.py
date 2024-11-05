#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 07:45:00 (ywatanabe)"
# File: ./mngs_repo/src/mngs/pd/_slice.py

from typing import Dict, Union, List

import pandas as pd

from ._find_indi import find_indi


def slice(df: pd.DataFrame, conditions: Dict[str, Union[str, int, float, List]]) -> pd.DataFrame:
    """Slices DataFrame rows that satisfy all given conditions.

    Example
    -------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'x']})
    >>> conditions = {'A': [1, 2], 'B': 'x'}
    >>> result = slice(df, conditions)
    >>> print(result)
       A  B
    0  1  x

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to slice
    conditions : Dict[str, Union[str, int, float, List]]
        Dictionary of column names and their target values

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rows that satisfy all conditions
    """
    return df[find_indi(df, conditions)].copy()

# EOF
