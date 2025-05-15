#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 10:15:25 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/test_io_consistency.py
# ----------------------------------------
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
import pytest
from typing import Any, Dict, List, Tuple, Callable
# ----------------------------------------

def test_roundtrip_numpy_array():
    """Test round-trip consistency for NumPy arrays."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test data
        original_data = np.array([1, 2, 3, 4, 5])
        
        # Paths for different formats
        npy_path = os.path.join(temp_dir, "test_array.npy")
        
        # Test NPY format
        _save(original_data, npy_path, verbose=False)
        loaded_data = load(npy_path)
        
        # Verify data integrity
        assert np.array_equal(original_data, loaded_data)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_roundtrip_numpy_matrix():
    """Test round-trip consistency for NumPy matrices."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test data
        original_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Paths for different formats
        npy_path = os.path.join(temp_dir, "test_matrix.npy")
        
        # Test NPY format
        _save(original_data, npy_path, verbose=False)
        loaded_data = load(npy_path)
        
        # Verify data integrity
        assert np.array_equal(original_data, loaded_data)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_roundtrip_dictionary():
    """Test round-trip consistency for dictionaries."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test data
        original_data = {"a": 1, "b": 2, "c": [3, 4, 5], "d": {"nested": "value"}}
        
        # Paths for different formats
        pkl_path = os.path.join(temp_dir, "test_dict.pkl")
        json_path = os.path.join(temp_dir, "test_dict.json")
        yaml_path = os.path.join(temp_dir, "test_dict.yaml")
        
        # Test PKL format
        _save(original_data, pkl_path, verbose=False)
        loaded_data_pkl = load(pkl_path)
        
        # Verify data integrity
        assert loaded_data_pkl == original_data
        
        # Test JSON format
        _save(original_data, json_path, verbose=False)
        loaded_data_json = load(json_path)
        
        # Verify data integrity (noting that JSON might convert some types)
        assert loaded_data_json["a"] == original_data["a"]
        assert loaded_data_json["b"] == original_data["b"]
        assert loaded_data_json["c"] == original_data["c"]
        assert loaded_data_json["d"]["nested"] == original_data["d"]["nested"]
        
        # Test YAML format
        _save(original_data, yaml_path, verbose=False)
        loaded_data_yaml = load(yaml_path)
        
        # Verify data integrity
        assert loaded_data_yaml["a"] == original_data["a"]
        assert loaded_data_yaml["b"] == original_data["b"]
        assert loaded_data_yaml["c"] == original_data["c"]
        assert loaded_data_yaml["d"]["nested"] == original_data["d"]["nested"]
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_roundtrip_list():
    """Test round-trip consistency for lists."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test data
        original_data = [1, 2, 3, "test", [4, 5, 6]]
        
        # Paths for different formats
        pkl_path = os.path.join(temp_dir, "test_list.pkl")
        
        # Test PKL format
        _save(original_data, pkl_path, verbose=False)
        loaded_data = load(pkl_path)
        
        # Verify data integrity
        assert loaded_data == original_data
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_roundtrip_torch_tensor():
    """Test round-trip consistency for PyTorch tensors."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test data
        original_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Paths for different formats
        pt_path = os.path.join(temp_dir, "test_tensor.pt")
        
        # Test PT format
        _save(original_data, pt_path, verbose=False)
        loaded_data = load(pt_path)
        
        # Verify data integrity
        assert torch.all(loaded_data == original_data)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_roundtrip_basic_dataframe():
    """Test round-trip consistency for basic Pandas DataFrames."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test data
        original_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [4.1, 5.2, 6.3]
        })
        
        # Paths for different formats
        csv_path = os.path.join(temp_dir, "test_df.csv")
        
        # Test CSV format
        _save(original_data, csv_path, verbose=False)
        loaded_data = load(csv_path)
        
        # Verify data integrity
        # For DataFrames we can use pandas testing utilities
        pd.testing.assert_frame_equal(
            loaded_data, 
            original_data,
            # Account for type changes in CSV roundtrip
            check_dtype=False
        )
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_multiple_save_load_cycles():
    """Test multiple save and load cycles to ensure idempotence."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test data
        original_data = np.array([1, 2, 3, 4, 5])
        
        # Path for test
        npy_path = os.path.join(temp_dir, "test_multi_cycle.npy")
        
        # First cycle
        _save(original_data, npy_path, verbose=False)
        loaded_data_1 = load(npy_path)
        
        # Second cycle
        _save(loaded_data_1, npy_path, verbose=False)
        loaded_data_2 = load(npy_path)
        
        # Third cycle
        _save(loaded_data_2, npy_path, verbose=False)
        loaded_data_3 = load(npy_path)
        
        # Verify data integrity across all cycles
        assert np.array_equal(original_data, loaded_data_1)
        assert np.array_equal(loaded_data_1, loaded_data_2)
        assert np.array_equal(loaded_data_2, loaded_data_3)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])