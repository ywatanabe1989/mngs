#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-03 07:53:00 (ywatanabe)"
# File: ./tests/mngs/stats/test__corr_test_multi.py

import pytest
import numpy as np
import pandas as pd


def test_corr_test_multi_with_dataframe():
    """Test corr_test_multi function with DataFrame input."""
    from mngs.stats._corr_test_multi import corr_test_multi
    
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'var1': np.random.randn(50),
        'var2': np.random.randn(50),
        'var3': np.random.randn(50)
    })
    
    # Test correlation
    result = corr_test_multi(data)
    
    # Check result properties
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)  # 3x3 correlation matrix
    
    # Check diagonal is 1 (self-correlation)
    assert np.allclose(np.diag(result.values), 1.0)
    
    # Check symmetry
    assert np.allclose(result.values, result.values.T)


def test_corr_test_multi_with_numpy_array():
    """Test corr_test_multi function with numpy array input."""
    from mngs.stats._corr_test_multi import corr_test_multi
    
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(50, 3)  # 50 samples, 3 variables
    
    # Test correlation
    result = corr_test_multi(data)
    
    # Check result properties
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)  # 3x3 correlation matrix
    
    # Check diagonal is 1 (self-correlation)
    assert np.allclose(np.diag(result.values), 1.0)


def test_corr_test_multi_perfect_correlation():
    """Test corr_test_multi with perfectly correlated data."""
    from mngs.stats._corr_test_multi import corr_test_multi
    
    # Create perfectly correlated data
    x = np.linspace(0, 10, 20)
    data = pd.DataFrame({
        'var1': x,
        'var2': 2 * x + 1,  # Perfect positive correlation
        'var3': -x + 5      # Perfect negative correlation
    })
    
    result = corr_test_multi(data)
    
    # Check perfect correlations
    assert abs(result.loc['var1', 'var2'] - 1.0) < 1e-10  # Should be 1
    assert abs(result.loc['var1', 'var3'] - (-1.0)) < 1e-10  # Should be -1


def test_corr_test_multi_single_variable():
    """Test corr_test_multi with single variable."""
    from mngs.stats._corr_test_multi import corr_test_multi
    
    # Single variable data
    data = pd.DataFrame({'var1': np.random.randn(20)})
    
    result = corr_test_multi(data)
    
    # Should be 1x1 matrix with value 1
    assert result.shape == (1, 1)
    assert result.iloc[0, 0] == 1.0


def test_corr_test_multi_with_constant_variable():
    """Test corr_test_multi with a constant variable."""
    from mngs.stats._corr_test_multi import corr_test_multi
    
    # Data with one constant variable
    data = pd.DataFrame({
        'var1': np.random.randn(20),
        'var2': np.ones(20),  # Constant
        'var3': np.random.randn(20)
    })
    
    # This should handle constant variables gracefully
    result = corr_test_multi(data)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)
    
    # Correlations with constant should be NaN or 0
    assert pd.isna(result.loc['var1', 'var2']) or result.loc['var1', 'var2'] == 0


def test_corr_test_multi_empty_data():
    """Test corr_test_multi with empty data."""
    from mngs.stats._corr_test_multi import corr_test_multi
    
    # Empty DataFrame
    empty_df = pd.DataFrame()
    
    # Should return empty correlation matrix
    result = corr_test_multi(empty_df)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (0, 0)


def test_corr_test_multi_with_missing_values():
    """Test corr_test_multi with missing values."""
    from mngs.stats._corr_test_multi import corr_test_multi
    
    # Data with NaN values
    data = pd.DataFrame({
        'var1': [1, 2, np.nan, 4, 5],
        'var2': [1, np.nan, 3, 4, 5],
        'var3': [1, 2, 3, 4, 5]
    })
    
    result = corr_test_multi(data)
    
    # Should handle NaN values (likely using pairwise complete observations)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)


def test_nocorrelation_test_basic():
    """Test nocorrelation_test function basic functionality."""
    from mngs.stats._corr_test_multi import nocorrelation_test
    
    # Create correlated data
    np.random.seed(42)
    x = np.random.randn(100)
    y = x + 0.5 * np.random.randn(100)
    
    result = nocorrelation_test(x, y)
    
    # Check result structure
    assert isinstance(result, dict)
    assert 'statistic' in result
    assert 'p_value' in result
    assert 'correlation' in result
    assert 'n' in result
    
    # Should show significant correlation
    assert result['p_value'] < 0.05
    assert result['correlation'] > 0.7
    assert result['n'] == 100


def test_nocorrelation_test_no_correlation():
    """Test nocorrelation_test with uncorrelated data."""
    from mngs.stats._corr_test_multi import nocorrelation_test
    
    # Create uncorrelated data
    np.random.seed(42)
    x = np.random.randn(100)
    y = np.random.randn(100)
    
    result = nocorrelation_test(x, y)
    
    # Should show no significant correlation
    assert result['p_value'] > 0.05
    assert abs(result['correlation']) < 0.3


def test_nocorrelation_test_perfect_correlation():
    """Test nocorrelation_test with perfect correlation."""
    from mngs.stats._corr_test_multi import nocorrelation_test
    
    x = np.array([1, 2, 3, 4, 5])
    y = 2 * x + 3
    
    result = nocorrelation_test(x, y)
    
    # Should show perfect correlation
    assert abs(result['correlation'] - 1.0) < 1e-10
    assert result['p_value'] < 0.001
    assert abs(result['statistic']) > 10  # Large t-statistic


def test_nocorrelation_test_with_nan():
    """Test nocorrelation_test with NaN values."""
    from mngs.stats._corr_test_multi import nocorrelation_test
    
    x = np.array([1, 2, np.nan, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    
    result = nocorrelation_test(x, y)
    
    # Should handle NaN by removing them
    assert result['n'] == 4  # One pair removed
    assert isinstance(result['correlation'], float)


def test_nocorrelation_test_t_statistic():
    """Test that t-statistic is calculated correctly."""
    from mngs.stats._corr_test_multi import nocorrelation_test
    
    np.random.seed(42)
    x = np.random.randn(50)
    y = x + np.random.randn(50)
    
    result = nocorrelation_test(x, y)
    
    # Verify t-statistic calculation
    r = result['correlation']
    n = result['n']
    expected_t = r * np.sqrt((n - 2) / (1 - r**2))
    
    assert abs(result['statistic'] - expected_t) < 1e-10


def test_corr_test_multi_correlation_values():
    """Test that correlation values are correct."""
    from mngs.stats._corr_test_multi import corr_test_multi
    from scipy import stats
    
    # Create test data with known correlations
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = x1 + 0.5 * np.random.randn(n)
    x3 = -x1 + 0.5 * np.random.randn(n)
    
    data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
    result = corr_test_multi(data)
    
    # Verify against scipy
    scipy_corr_12, _ = stats.pearsonr(x1, x2)
    scipy_corr_13, _ = stats.pearsonr(x1, x3)
    
    assert abs(result.loc['x1', 'x2'] - scipy_corr_12) < 1e-10
    assert abs(result.loc['x1', 'x3'] - scipy_corr_13) < 1e-10


if __name__ == "__main__":
    import os
    pytest.main([os.path.abspath(__file__)])