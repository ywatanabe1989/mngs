#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 11:10:45 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/test_nested_structures_consistency.py
# ----------------------------------------
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest
from typing import Any, Dict, List, Tuple
# ----------------------------------------

def test_dict_of_dataframes():
    """Test preservation of dictionaries containing DataFrames."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create dictionary of DataFrames
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df2 = pd.DataFrame({'X': [7, 8, 9], 'Y': [10, 11, 12]})
        df3 = pd.DataFrame({'M': [13, 14], 'N': [15, 16]})
        
        data_dict = {
            'first_df': df1,
            'second_df': df2,
            'third_df': df3
        }
        
        # Test path
        pkl_path = os.path.join(temp_dir, "dict_of_dfs.pkl")
        
        # Save and load
        _save(data_dict, pkl_path, verbose=False)
        loaded_dict = load(pkl_path)
        
        # Verify structure is preserved
        assert set(loaded_dict.keys()) == set(data_dict.keys())
        
        # Verify each DataFrame is preserved
        pd.testing.assert_frame_equal(loaded_dict['first_df'], df1)
        pd.testing.assert_frame_equal(loaded_dict['second_df'], df2)
        pd.testing.assert_frame_equal(loaded_dict['third_df'], df3)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_list_of_dataframes():
    """Test preservation of lists containing DataFrames."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create list of DataFrames
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df2 = pd.DataFrame({'X': [7, 8, 9], 'Y': [10, 11, 12]})
        df3 = pd.DataFrame({'M': [13, 14], 'N': [15, 16]})
        
        data_list = [df1, df2, df3]
        
        # Test path
        pkl_path = os.path.join(temp_dir, "list_of_dfs.pkl")
        
        # Save and load
        _save(data_list, pkl_path, verbose=False)
        loaded_list = load(pkl_path)
        
        # Verify structure is preserved
        assert len(loaded_list) == len(data_list)
        
        # Verify each DataFrame is preserved
        pd.testing.assert_frame_equal(loaded_list[0], df1)
        pd.testing.assert_frame_equal(loaded_list[1], df2)
        pd.testing.assert_frame_equal(loaded_list[2], df3)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_nested_dict_structure():
    """Test preservation of deeply nested dictionary structures."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create nested dictionary with various data types
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        array1 = np.array([5, 6, 7])
        
        nested_dict = {
            'level1': {
                'level2a': {
                    'dataframe': df1,
                    'array': array1,
                    'scalar': 42
                },
                'level2b': {
                    'string': 'test',
                    'list': [1, 2, 3],
                    'boolean': True
                }
            },
            'simple_key': 'simple_value'
        }
        
        # Test paths for different formats
        pkl_path = os.path.join(temp_dir, "nested_dict.pkl")
        
        # Save and load with pickle (best for preserving complex structures)
        _save(nested_dict, pkl_path, verbose=False)
        loaded_dict = load(pkl_path)
        
        # Verify structure is preserved
        assert set(loaded_dict.keys()) == set(nested_dict.keys())
        assert set(loaded_dict['level1'].keys()) == set(nested_dict['level1'].keys())
        assert set(loaded_dict['level1']['level2a'].keys()) == set(nested_dict['level1']['level2a'].keys())
        assert set(loaded_dict['level1']['level2b'].keys()) == set(nested_dict['level1']['level2b'].keys())
        
        # Verify DataFrame
        pd.testing.assert_frame_equal(
            loaded_dict['level1']['level2a']['dataframe'], 
            nested_dict['level1']['level2a']['dataframe']
        )
        
        # Verify array
        np.testing.assert_array_equal(
            loaded_dict['level1']['level2a']['array'],
            nested_dict['level1']['level2a']['array']
        )
        
        # Verify scalar values
        assert loaded_dict['level1']['level2a']['scalar'] == nested_dict['level1']['level2a']['scalar']
        assert loaded_dict['level1']['level2b']['string'] == nested_dict['level1']['level2b']['string']
        assert loaded_dict['level1']['level2b']['list'] == nested_dict['level1']['level2b']['list']
        assert loaded_dict['level1']['level2b']['boolean'] == nested_dict['level1']['level2b']['boolean']
        assert loaded_dict['simple_key'] == nested_dict['simple_key']
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_dataframe_with_nested_objects():
    """Test preservation of DataFrames containing complex/nested objects."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create lists and dicts to store in DataFrame
        list1 = [1, 2, 3]
        list2 = [4, 5, 6]
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'c': 3, 'd': 4}
        
        # Create DataFrame with complex objects
        df = pd.DataFrame({
            'lists': [list1, list2],
            'dicts': [dict1, dict2],
            'scalar': [10, 20]
        })
        
        # Test path
        pkl_path = os.path.join(temp_dir, "df_with_nested.pkl")
        
        # Save and load with pickle
        _save(df, pkl_path, verbose=False)
        loaded_df = load(pkl_path)
        
        # Verify nested objects are preserved
        assert loaded_df['lists'][0] == list1
        assert loaded_df['lists'][1] == list2
        assert loaded_df['dicts'][0] == dict1
        assert loaded_df['dicts'][1] == dict2
        assert loaded_df['scalar'][0] == 10
        assert loaded_df['scalar'][1] == 20
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_numpy_structured_array():
    """Test preservation of numpy structured arrays."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create structured array
        dtype = [('name', 'U10'), ('height', float), ('age', int)]
        values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38), ('Galahad', 1.7, 25)]
        structured_array = np.array(values, dtype=dtype)
        
        # Test path
        npy_path = os.path.join(temp_dir, "structured_array.npy")
        
        # Save and load
        _save(structured_array, npy_path, verbose=False)
        loaded_array = load(npy_path)
        
        # Verify structured array is preserved
        assert loaded_array.dtype == structured_array.dtype
        np.testing.assert_array_equal(loaded_array, structured_array)
        
        # Verify field access still works
        assert all(loaded_array['name'] == structured_array['name'])
        assert all(loaded_array['height'] == structured_array['height'])
        assert all(loaded_array['age'] == structured_array['age'])
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_npz_dict_preservation():
    """Test preservation of dictionaries saved as NPZ files."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create dictionary of arrays
        arrays_dict = {
            'arr1': np.array([1, 2, 3]),
            'arr2': np.array([[4, 5], [6, 7]]),
            'arr3': np.array([8.1, 9.2, 10.3])
        }
        
        # Test path
        npz_path = os.path.join(temp_dir, "arrays_dict.npz")
        
        # Save and load
        _save(arrays_dict, npz_path, verbose=False)
        loaded_arrays = load(npz_path)
        
        # NPZ files load as a special NpzFile object, but we can convert to list
        loaded_list = loaded_arrays  # Should already be a list based on _load_npz implementation
        
        # Verify the arrays are preserved (note: keys are lost in the NPZ loading process)
        # Instead, we check the arrays match by value, regardless of order
        original_arrays = list(arrays_dict.values())
        
        # Sort arrays by their first element to make comparison easier
        original_arrays.sort(key=lambda x: x.flatten()[0])
        loaded_list.sort(key=lambda x: x.flatten()[0])
        
        for i in range(len(original_arrays)):
            np.testing.assert_array_equal(loaded_list[i], original_arrays[i])
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])