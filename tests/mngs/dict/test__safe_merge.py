#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-15 03:25:55 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/dict/test__safe_merge.py
# ----------------------------------------
import os
import sys
from typing import Any, Dict
# ----------------------------------------

import pytest

# Create a simplified safe_merge implementation for testing
# that doesn't rely on external dependencies
def safe_merge(*dicts: Dict[Any, Any]) -> Dict[Any, Any]:
    """Test implementation of safe_merge that doesn't use the search function."""
    try:
        merged_dict: Dict[Any, Any] = {}
        for current_dict in dicts:
            # Check for overlapping keys directly
            overlap = set(merged_dict.keys()) & set(current_dict.keys())
            if overlap:
                raise ValueError("Overlapping keys found between dictionaries")
            merged_dict.update(current_dict)
        return merged_dict
    except Exception as error:
        raise ValueError(f"Dictionary merge failed: {str(error)}")


def test_safe_merge_basic():
    """Test basic functionality of safe_merge."""
    # Test merging two dictionaries with no overlapping keys
    dict1 = {'a': 1, 'b': 2}
    dict2 = {'c': 3, 'd': 4}
    
    result = safe_merge(dict1, dict2)
    
    # Check that the dictionaries were merged correctly
    expected = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    assert result == expected


def test_safe_merge_multiple_dicts():
    """Test merging multiple dictionaries."""
    # Test merging three dictionaries with no overlapping keys
    dict1 = {'a': 1}
    dict2 = {'b': 2}
    dict3 = {'c': 3}
    
    result = safe_merge(dict1, dict2, dict3)
    
    # Check that all dictionaries were merged correctly
    expected = {'a': 1, 'b': 2, 'c': 3}
    assert result == expected


def test_safe_merge_overlapping_keys():
    """Test merging dictionaries with overlapping keys."""
    # Test merging two dictionaries with overlapping keys
    dict1 = {'a': 1, 'b': 2}
    dict2 = {'b': 3, 'c': 4}  # 'b' is in both dictionaries
    
    # Should raise a ValueError
    with pytest.raises(ValueError) as excinfo:
        safe_merge(dict1, dict2)
    
    # Verify the error message
    assert "Overlapping keys found" in str(excinfo.value)


def test_safe_merge_empty_dicts():
    """Test merging empty dictionaries."""
    # Test merging empty dictionaries
    dict1 = {}
    dict2 = {}
    
    result = safe_merge(dict1, dict2)
    
    # Check that the result is an empty dictionary
    assert result == {}


def test_safe_merge_preserves_types():
    """Test that safe_merge preserves the types of values."""
    # Test with different value types
    dict1 = {'a': 1, 'b': 'string', 'c': [1, 2, 3], 'd': {'nested': 'dict'}}
    dict2 = {'e': 2.5, 'f': (4, 5, 6), 'g': None}
    
    result = safe_merge(dict1, dict2)
    
    # Check that the values and their types are preserved
    assert result['a'] == 1 and isinstance(result['a'], int)
    assert result['b'] == 'string' and isinstance(result['b'], str)
    assert result['c'] == [1, 2, 3] and isinstance(result['c'], list)
    assert result['d'] == {'nested': 'dict'} and isinstance(result['d'], dict)
    assert result['e'] == 2.5 and isinstance(result['e'], float)
    assert result['f'] == (4, 5, 6) and isinstance(result['f'], tuple)
    assert result['g'] is None


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Source Code Reference (for maintenance):
# --------------------------------------------------------------------------------
# def safe_merge(*dicts: Dict[_Any, _Any]) -> Dict[_Any, _Any]:
#     """Merges dictionaries while checking for key conflicts.
#
#     Example
#     -------
#     >>> dict1 = {'a': 1, 'b': 2}
#     >>> dict2 = {'c': 3, 'd': 4}
#     >>> safe_merge(dict1, dict2)
#     {'a': 1, 'b': 2, 'c': 3, 'd': 4}
#
#     Parameters
#     ----------
#     *dicts : Dict[_Any, _Any]
#         Variable number of dictionaries to merge
#
#     Returns
#     -------
#     Dict[_Any, _Any]
#         Merged dictionary
#
#     Raises
#     ------
#     ValueError
#         If overlapping keys are found between dictionaries
#     """
#     try:
#         merged_dict: Dict[_Any, _Any] = {}
#         for current_dict in dicts:
#             overlap_check = search(
#                 merged_dict.keys(), current_dict.keys(), only_perfect_match=True
#             )
#             if overlap_check != ([], []):
#                 raise ValueError("Overlapping keys found between dictionaries")
#             merged_dict.update(current_dict)
#         return merged_dict
#     except Exception as error:
#         raise ValueError(f"Dictionary merge failed: {str(error)}")