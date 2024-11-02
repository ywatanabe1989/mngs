#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 04:08:45 (ywatanabe)"
# File: ./mngs_repo/src/mngs/dict/_safe_merge.py

"""
Functionality:
    - Safely merges multiple dictionaries without overlapping keys
Input:
    - Multiple dictionaries to be merged
Output:
    - A single merged dictionary
Prerequisites:
    - mngs.general package with search function
"""

from typing import Dict, Any
from ..general import search

def safe_merge(*dicts: Dict[Any, Any]) -> Dict[Any, Any]:
    """Merges dictionaries while checking for key conflicts.

    Example
    -------
    >>> dict1 = {'a': 1, 'b': 2}
    >>> dict2 = {'c': 3, 'd': 4}
    >>> safe_merge(dict1, dict2)
    {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    Parameters
    ----------
    *dicts : Dict[Any, Any]
        Variable number of dictionaries to merge

    Returns
    -------
    Dict[Any, Any]
        Merged dictionary

    Raises
    ------
    ValueError
        If overlapping keys are found between dictionaries
    """
    try:
        merged_dict: Dict[Any, Any] = {}
        for current_dict in dicts:
            overlap_check = search(
                merged_dict.keys(),
                current_dict.keys(),
                only_perfect_match=True
            )
            if overlap_check != ([], []):
                raise ValueError("Overlapping keys found between dictionaries")
            merged_dict.update(current_dict)
        return merged_dict
    except Exception as error:
        raise ValueError(f"Dictionary merge failed: {str(error)}")

# EOF
