#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 10:50:22 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/test_special_datatypes_consistency.py
# ----------------------------------------
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
import pytz
from typing import Any, Dict, List, Tuple
# ----------------------------------------

def test_datetime_preservation():
    """Test preservation of datetime values in DataFrames."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create DataFrame with datetime data
        base_date = datetime(2025, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(5)]
        
        df = pd.DataFrame({
            'date': dates,
            'value': range(1, 6)
        })
        
        # Test paths for different formats
        csv_path = os.path.join(temp_dir, "datetime_test.csv")
        pkl_path = os.path.join(temp_dir, "datetime_test.pkl")
        
        # Test CSV format - note that CSV might convert to string
        _save(df, csv_path, verbose=False)
        csv_loaded = load(csv_path)
        
        # Convert loaded dates back to datetime for comparison
        csv_loaded['date'] = pd.to_datetime(csv_loaded['date'])
        
        # Compare values
        assert all(csv_loaded['date'].dt.date == pd.Series(dates).dt.date)
        
        # Test pickle format which should preserve types perfectly
        _save(df, pkl_path, verbose=False)
        pkl_loaded = load(pkl_path)
        
        # Verify datetime preservation with pickle
        pd.testing.assert_frame_equal(df, pkl_loaded)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_timezone_datetime_preservation():
    """Test preservation of timezone-aware datetime values."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create timezone-aware datetimes
        base_date = datetime(2025, 1, 1, tzinfo=pytz.UTC)
        dates = [base_date + timedelta(days=i) for i in range(5)]
        
        # Create test data
        df = pd.DataFrame({
            'tz_date': dates,
            'value': range(1, 6)
        })
        
        # Test path
        pkl_path = os.path.join(temp_dir, "tz_datetime_test.pkl")
        
        # Save and load using pickle (most reliable for complex types)
        _save(df, pkl_path, verbose=False)
        loaded_df = load(pkl_path)
        
        # Verify timezone information is preserved
        for i, dt in enumerate(loaded_df['tz_date']):
            assert dt.tzinfo is not None
            assert dt.tzinfo.tzname(dt) == 'UTC'
            assert dt == dates[i]
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_categorical_data_preservation():
    """Test preservation of categorical data types."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create DataFrame with categorical column
        categories = ['Low', 'Medium', 'High']
        cat_data = pd.Series(['Medium', 'Low', 'High', 'Medium', 'High'], 
                           dtype=pd.CategoricalDtype(categories=categories, ordered=True))
        
        df = pd.DataFrame({
            'category': cat_data,
            'value': range(1, 6)
        })
        
        # Test paths
        pkl_path = os.path.join(temp_dir, "categorical_test.pkl")
        
        # Save and load with pickle (preserves types)
        _save(df, pkl_path, verbose=False)
        loaded_df = load(pkl_path)
        
        # Verify categorical type preservation
        assert pd.api.types.is_categorical_dtype(loaded_df['category'])
        assert loaded_df['category'].cat.ordered
        assert list(loaded_df['category'].cat.categories) == categories
        
        # Verify values
        assert all(loaded_df['category'] == df['category'])
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_nullable_integer_preservation():
    """Test preservation of nullable integer types."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create DataFrame with nullable integer column (pandas Int64)
        df = pd.DataFrame({
            'int_col': pd.Series([1, 2, None, 4, 5], dtype='Int64'),
            'value': range(1, 6)
        })
        
        # Test paths
        pkl_path = os.path.join(temp_dir, "nullable_int_test.pkl")
        
        # Save and load with pickle
        _save(df, pkl_path, verbose=False)
        loaded_df = load(pkl_path)
        
        # Verify nullable integer type preservation
        assert pd.api.types.is_integer_dtype(loaded_df['int_col'])
        assert loaded_df['int_col'].isna().sum() == 1  # Check None is preserved
        
        # Compare values
        pd.testing.assert_series_equal(
            loaded_df['int_col'].fillna(0),
            df['int_col'].fillna(0)
        )
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_complex_number_preservation():
    """Test preservation of complex numbers."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create array with complex numbers
        complex_array = np.array([1+2j, 3+4j, 5+6j, 7+8j])
        
        # Test paths
        npy_path = os.path.join(temp_dir, "complex_test.npy")
        
        # Save and load with numpy
        _save(complex_array, npy_path, verbose=False)
        loaded_array = load(npy_path)
        
        # Verify complex number preservation
        assert loaded_array.dtype == np.complex128
        np.testing.assert_array_equal(loaded_array, complex_array)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_boolean_preservation():
    """Test preservation of boolean data types."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create DataFrame with boolean column
        df = pd.DataFrame({
            'bool_col': [True, False, True, False, True],
            'value': range(1, 6)
        })
        
        # Test paths
        csv_path = os.path.join(temp_dir, "boolean_test.csv")
        pkl_path = os.path.join(temp_dir, "boolean_test.pkl")
        
        # Test with CSV
        _save(df, csv_path, verbose=False)
        csv_loaded = load(csv_path)
        
        # With CSV, booleans might be converted to integer (1/0)
        # or strings ('True'/'False'), so we check values not types
        assert all((csv_loaded['bool_col'] == 1) == df['bool_col'])
        
        # Test with pickle (should preserve types)
        _save(df, pkl_path, verbose=False)
        pkl_loaded = load(pkl_path)
        
        # Verify boolean type preservation
        assert pd.api.types.is_bool_dtype(pkl_loaded['bool_col'])
        assert all(pkl_loaded['bool_col'] == df['bool_col'])
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_nan_none_preservation():
    """Test preservation of NaN and None values."""
    from mngs.io._save import _save
    from mngs.io._load import load
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create DataFrame with NaN and None values
        df = pd.DataFrame({
            'with_nan': [1.0, np.nan, 3.0, np.nan, 5.0],
            'with_none': [1, None, 3, None, 5],
            'with_both': [np.nan, None, 3, 4, 5]
        })
        
        # Test paths
        pkl_path = os.path.join(temp_dir, "nan_none_test.pkl")
        
        # Save and load with pickle
        _save(df, pkl_path, verbose=False)
        loaded_df = load(pkl_path)
        
        # Check NaN values (use isna() since direct comparison with NaN doesn't work)
        assert loaded_df['with_nan'].isna().sum() == 2
        assert loaded_df['with_none'].isna().sum() == 2
        assert loaded_df['with_both'].isna().sum() == 2
        
        # Check pattern of NA values
        pd.testing.assert_series_equal(
            loaded_df['with_nan'].isna(),
            df['with_nan'].isna()
        )
        pd.testing.assert_series_equal(
            loaded_df['with_none'].isna(),
            df['with_none'].isna()
        )
        pd.testing.assert_series_equal(
            loaded_df['with_both'].isna(),
            df['with_both'].isna()
        )
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])