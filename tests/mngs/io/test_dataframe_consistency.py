#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 10:32:15 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/test_dataframe_consistency.py
# ----------------------------------------
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest
from typing import Any, Dict, List, Tuple
# ----------------------------------------

def test_dataframe_index_preservation():
    """Test preservation of DataFrame index structures across save/load cycles."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test DataFrame with custom index
        df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': ['a', 'b', 'c', 'd'],
        }, index=pd.Index(['w', 'x', 'y', 'z'], name='custom_index'))
        
        # Test path
        csv_path = os.path.join(temp_dir, "test_index.csv")
        
        # Save and load
        _save(df, csv_path, verbose=False)
        loaded_df = load(csv_path)
        
        # Verify index structure is preserved
        assert loaded_df.index.name == df.index.name
        assert all(loaded_df.index == df.index)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_dataframe_multiindex_preservation():
    """Test preservation of DataFrame MultiIndex across save/load cycles."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test DataFrame with MultiIndex
        index = pd.MultiIndex.from_tuples(
            [('A', 1), ('A', 2), ('B', 1), ('B', 2)],
            names=['letter', 'number']
        )
        df = pd.DataFrame({
            'value1': [1, 2, 3, 4],
            'value2': [5, 6, 7, 8]
        }, index=index)
        
        # Test path
        csv_path = os.path.join(temp_dir, "test_multiindex.csv")
        
        # Save and load
        _save(df, csv_path, verbose=False)
        loaded_df = load(csv_path)
        
        # Verify MultiIndex structure is preserved
        assert isinstance(loaded_df.index, pd.MultiIndex)
        assert loaded_df.index.names == df.index.names
        assert loaded_df.index.levshape == df.index.levshape
        
        # Verify values
        pd.testing.assert_frame_equal(
            loaded_df, 
            df,
            check_dtype=False  # CSV may change data types
        )
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_dataframe_column_order_preservation():
    """Test preservation of DataFrame column order across save/load cycles."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test DataFrame with specific column order
        df = pd.DataFrame({
            'C': [7, 8, 9],
            'A': [1, 2, 3],
            'D': [10, 11, 12],
            'B': [4, 5, 6],
        })
        
        # Ensure specific column order
        df = df[['C', 'A', 'D', 'B']]
        original_columns = list(df.columns)
        
        # Test path
        csv_path = os.path.join(temp_dir, "test_column_order.csv")
        
        # Save and load
        _save(df, csv_path, verbose=False)
        loaded_df = load(csv_path)
        
        # Verify column order is preserved
        assert list(loaded_df.columns) == original_columns
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_dataframe_multiindex_columns_preservation():
    """Test preservation of MultiIndex columns across save/load cycles."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test DataFrame with MultiIndex columns
        columns = pd.MultiIndex.from_tuples(
            [('A', 'one'), ('A', 'two'), ('B', 'one'), ('B', 'two')],
            names=['first', 'second']
        )
        df = pd.DataFrame(
            np.random.rand(3, 4),
            index=['X', 'Y', 'Z'],
            columns=columns
        )
        
        # Test path
        csv_path = os.path.join(temp_dir, "test_multiindex_columns.csv")
        
        # Save and load
        _save(df, csv_path, verbose=False)
        loaded_df = load(csv_path)
        
        # Verify MultiIndex column structure is preserved
        assert isinstance(loaded_df.columns, pd.MultiIndex)
        assert loaded_df.columns.names == df.columns.names
        assert loaded_df.columns.levshape == df.columns.levshape
        
        # Verify data with approximate comparison due to floating point values
        np.testing.assert_array_almost_equal(
            loaded_df.values,
            df.values
        )
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_dataframe_named_columns_preservation():
    """Test preservation of named columns across save/load cycles."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test DataFrame with named columns
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        df.columns.name = 'variables'
        
        # Test path
        csv_path = os.path.join(temp_dir, "test_named_columns.csv")
        
        # Save and load
        _save(df, csv_path, verbose=False)
        loaded_df = load(csv_path)
        
        # Verify column name is preserved
        assert loaded_df.columns.name == df.columns.name
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_dataframe_empty_preservation():
    """Test preservation of empty DataFrames across save/load cycles."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create empty DataFrame with columns
        df = pd.DataFrame(columns=['A', 'B', 'C'])
        
        # Test path
        csv_path = os.path.join(temp_dir, "test_empty_df.csv")
        
        # Save and load
        _save(df, csv_path, verbose=False)
        loaded_df = load(csv_path)
        
        # Verify empty structure is preserved
        assert loaded_df.empty
        assert list(loaded_df.columns) == list(df.columns)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_dataframe_roundtrip_multiple_formats():
    """Test DataFrame consistency across multiple save/load cycles with different formats."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test DataFrame
        original_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [4.1, 5.2, 6.3]
        }, index=pd.Index(['x', 'y', 'z'], name='idx'))
        
        # Test paths for different formats
        csv_path = os.path.join(temp_dir, "df_cycle.csv")
        pkl_path = os.path.join(temp_dir, "df_cycle.pkl")
        
        # First cycle: Original → CSV → Load
        _save(original_df, csv_path, verbose=False)
        csv_df = load(csv_path)
        
        # Second cycle: CSV → PKL → Load
        _save(csv_df, pkl_path, verbose=False)
        pkl_df = load(pkl_path)
        
        # Third cycle: PKL → CSV → Load
        _save(pkl_df, csv_path, verbose=False)
        final_df = load(csv_path)
        
        # Verify consistency across all cycles (ignoring data types due to CSV conversions)
        pd.testing.assert_frame_equal(original_df, csv_df, check_dtype=False)
        pd.testing.assert_frame_equal(csv_df, pkl_df, check_dtype=False)
        pd.testing.assert_frame_equal(pkl_df, final_df, check_dtype=False)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])