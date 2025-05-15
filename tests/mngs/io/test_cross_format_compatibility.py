#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 11:30:28 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/test_cross_format_compatibility.py
# ----------------------------------------
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest
import torch
from typing import Any, Dict, List, Tuple
# ----------------------------------------

def test_numeric_data_cross_format():
    """Test cross-format compatibility for basic numeric data."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test numeric data
        original_data = np.array([1, 2, 3, 4, 5])
        
        # Save in different formats
        npy_path = os.path.join(temp_dir, "data.npy")
        csv_path = os.path.join(temp_dir, "data.csv")
        pkl_path = os.path.join(temp_dir, "data.pkl")
        json_path = os.path.join(temp_dir, "data.json")
        
        # Save to all formats
        _save(original_data, npy_path, verbose=False)
        _save(original_data, csv_path, verbose=False)
        _save(original_data, pkl_path, verbose=False)
        _save(original_data.tolist(), json_path, verbose=False)  # Convert to list for JSON
        
        # Load from all formats
        npy_data = load(npy_path)
        csv_data = load(csv_path)
        pkl_data = load(pkl_path)
        json_data = load(json_path)
        
        # Verify base values across formats (ignoring type differences)
        np.testing.assert_array_equal(npy_data, original_data)
        
        # CSV may load as DataFrame, so extract values
        if isinstance(csv_data, pd.DataFrame):
            # This may need adjustment based on how the CSV is structured
            csv_values = csv_data.iloc[:, 0].values
            np.testing.assert_array_equal(csv_values, original_data)
        
        np.testing.assert_array_equal(pkl_data, original_data)
        assert json_data == original_data.tolist()
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_dataframe_cross_format():
    """Test cross-format compatibility for DataFrames."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test DataFrame
        original_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        
        # Save in different formats
        csv_path = os.path.join(temp_dir, "data.csv")
        pkl_path = os.path.join(temp_dir, "data.pkl")
        
        # Save to formats
        _save(original_df, csv_path, verbose=False)
        _save(original_df, pkl_path, verbose=False)
        
        # Load from formats
        csv_df = load(csv_path)
        pkl_df = load(pkl_path)
        
        # Convert loaded data to DataFrames if needed
        if not isinstance(csv_df, pd.DataFrame):
            csv_df = pd.DataFrame(csv_df)
        
        if not isinstance(pkl_df, pd.DataFrame):
            pkl_df = pd.DataFrame(pkl_df)
        
        # Compare data values
        pd.testing.assert_frame_equal(
            csv_df.sort_index(axis=1).reset_index(drop=True),
            original_df.sort_index(axis=1).reset_index(drop=True),
            check_dtype=False
        )
        
        pd.testing.assert_frame_equal(
            pkl_df.sort_index(axis=1).reset_index(drop=True),
            original_df.sort_index(axis=1).reset_index(drop=True)
        )
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_dict_cross_format():
    """Test cross-format compatibility for dictionaries."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create simple dictionary
        original_dict = {
            'a': 1,
            'b': 2,
            'c': "string",
            'd': [3, 4, 5]
        }
        
        # Save in different formats
        pkl_path = os.path.join(temp_dir, "dict.pkl")
        json_path = os.path.join(temp_dir, "dict.json")
        yaml_path = os.path.join(temp_dir, "dict.yaml")
        
        # Save to formats
        _save(original_dict, pkl_path, verbose=False)
        _save(original_dict, json_path, verbose=False)
        _save(original_dict, yaml_path, verbose=False)
        
        # Load from formats
        pkl_dict = load(pkl_path)
        json_dict = load(json_path)
        yaml_dict = load(yaml_path)
        
        # Verify loaded data
        assert pkl_dict == original_dict
        assert json_dict == original_dict
        assert yaml_dict == original_dict
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_numpy_to_pytorch_compatibility():
    """Test compatibility between NumPy and PyTorch for tensors."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create NumPy array
        np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        
        # Convert to PyTorch tensor
        torch_tensor = torch.tensor(np_array)
        
        # Paths for different formats
        npy_path = os.path.join(temp_dir, "data.npy")
        pt_path = os.path.join(temp_dir, "data.pt")
        
        # Test NumPy → PT conversion
        _save(np_array, npy_path, verbose=False)
        _save(torch_tensor, pt_path, verbose=False)
        
        # Load with opposite format
        np_from_pt = load(pt_path)
        torch_from_npy = torch.tensor(load(npy_path))
        
        # Verify data
        np.testing.assert_array_equal(np_from_pt.numpy(), np_array)
        assert torch.all(torch_from_npy == torch_tensor)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_pandas_to_numpy_compatibility():
    """Test compatibility between Pandas and NumPy data formats."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create Pandas DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        # Create equivalent NumPy array
        np_array = df.values
        
        # Paths for different formats
        csv_path = os.path.join(temp_dir, "data.csv")
        npy_path = os.path.join(temp_dir, "data.npy")
        
        # Save formats
        _save(df, csv_path, verbose=False)
        _save(np_array, npy_path, verbose=False)
        
        # Load cross-format
        df_from_npy = pd.DataFrame(load(npy_path))
        np_from_csv = load(csv_path).values
        
        # Verify data
        pd.testing.assert_frame_equal(
            df_from_npy,
            pd.DataFrame(np_array),
            check_dtype=False,
            check_index_type=False,
            check_column_type=False
        )
        
        np.testing.assert_array_equal(np_from_csv, np_array)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_list_conversion_compatibility():
    """Test compatible conversions between lists and other formats."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create original list
        original_list = [1, 2, 3, 4, 5]
        
        # Paths for different formats
        json_path = os.path.join(temp_dir, "list.json")
        yaml_path = os.path.join(temp_dir, "list.yaml")
        pkl_path = os.path.join(temp_dir, "list.pkl")
        npy_path = os.path.join(temp_dir, "list.npy")
        
        # Save in different formats
        _save(original_list, json_path, verbose=False)
        _save(original_list, yaml_path, verbose=False)
        _save(original_list, pkl_path, verbose=False)
        _save(np.array(original_list), npy_path, verbose=False)
        
        # Load from different formats
        json_data = load(json_path)
        yaml_data = load(yaml_path)
        pkl_data = load(pkl_path)
        npy_data = load(npy_path)
        
        # Verify base values (direct comparison for non-numpy)
        assert json_data == original_list
        assert yaml_data == original_list
        assert pkl_data == original_list
        
        # For NumPy, convert back to list for comparison
        assert npy_data.tolist() == original_list
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_multi_format_roundtrip():
    """Test data integrity through multiple format conversions."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create original data
        original_data = {
            'values': [1, 2, 3, 4, 5],
            'name': 'test_data',
            'metadata': {
                'created': '2025-05-16',
                'version': 1.0
            }
        }
        
        # Define conversion path through multiple formats
        # dict -> json -> yaml -> pkl -> dict
        json_path = os.path.join(temp_dir, "stage1.json")
        yaml_path = os.path.join(temp_dir, "stage2.yaml")
        pkl_path = os.path.join(temp_dir, "stage3.pkl")
        
        # Execute conversion path
        _save(original_data, json_path, verbose=False)
        json_data = load(json_path)
        
        _save(json_data, yaml_path, verbose=False)
        yaml_data = load(yaml_path)
        
        _save(yaml_data, pkl_path, verbose=False)
        final_data = load(pkl_path)
        
        # Verify data integrity after full roundtrip
        assert final_data == original_data
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])