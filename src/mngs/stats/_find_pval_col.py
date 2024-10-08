#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-06 09:38:04 (ywatanabe)"
# /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/stats/_find_pval_col.py

"""
Functionality:
    - Identifies the column name in a DataFrame that corresponds to p-values
Input:
    - pandas DataFrame, array-like object, list, or dict
Output:
    - String representing the identified p-value column name, or None if not found
Prerequisites:
    - pandas library
"""

import re
from typing import Optional, Union, List, Dict
import pandas as pd
import numpy as np

def find_pval(data: Union[pd.DataFrame, np.ndarray, List, Dict]) -> Optional[str]:
    """
    Find the column name or key that matches p-value patterns.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'p_value': [0.05, 0.01], 'pval': [0.1, 0.001], 'other': [1, 2]})
    >>> find_pval(df)
    'p_value'

    Parameters:
    -----------
    data : Union[pd.DataFrame, np.ndarray, List, Dict]
        Data structure to search for p-value column or key

    Returns:
    --------
    Optional[str]
        Name of the column or key that matches p-value patterns, or None if not found
    """
    if isinstance(data, pd.DataFrame):
        return find_pval_col(data)
    elif isinstance(data, (np.ndarray, list, dict)):
        return _find_pval(data)
    else:
        raise ValueError("Input must be a pandas DataFrame, numpy array, list, or dict")

def _find_pval(data: Union[np.ndarray, List, Dict]) -> Optional[str]:
    pattern = re.compile(r'p[-_]?val(ue)?', re.IGNORECASE)
    if isinstance(data, dict):
        for key in data.keys():
            if pattern.search(str(key)):
                return key
    elif isinstance(data, (np.ndarray, list)):
        if len(data) > 0 and isinstance(data[0], dict):
            for key in data[0].keys():
                if pattern.search(str(key)):
                    return key
    return None

def find_pval_col(df: pd.DataFrame) -> Optional[str]:
    """
    Find the column name that matches p-value patterns in a DataFrame.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'p_value': [0.05, 0.01], 'pval': [0.1, 0.001], 'other': [1, 2]})
    >>> find_pval_col(df)
    'p_value'

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to search for p-value column

    Returns:
    --------
    Optional[str]
        Name of the column that matches p-value patterns, or None if not found
    """
    pattern = re.compile(r'p[-_]?val(ue)?', re.IGNORECASE)
    for col in df.columns:
        if pattern.search(str(col)):
            return col
    return None
