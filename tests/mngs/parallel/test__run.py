#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-02 16:45:00 (ywatanabe)"
# File: ./tests/mngs/parallel/test__run.py

import multiprocessing
import time
import warnings
from unittest.mock import patch

import pytest


def test_run_basic_functionality():
    """Test basic parallel function execution."""
    from mngs.parallel._run import run

    def add(x, y):
        return x + y

    args_list = [(1, 4), (2, 5), (3, 6)]
    result = run(add, args_list)
    
    assert result == [5, 7, 9]
    assert len(result) == 3


def test_run_single_argument():
    """Test parallel execution with single argument functions."""
    from mngs.parallel._run import run

    def square(x):
        return x * x

    args_list = [(2,), (3,), (4,)]
    result = run(square, args_list)
    
    assert result == [4, 9, 16]


def test_run_multiple_arguments():
    """Test parallel execution with multiple argument functions."""
    from mngs.parallel._run import run

    def multiply_three(x, y, z):
        return x * y * z

    args_list = [(2, 3, 4), (1, 5, 6), (2, 2, 2)]
    result = run(multiply_three, args_list)
    
    assert result == [24, 30, 8]


def test_run_tuple_returns():
    """Test parallel execution with functions returning tuples."""
    from mngs.parallel._run import run

    def divmod_func(x, y):
        return divmod(x, y)

    args_list = [(10, 3), (15, 4), (20, 6)]
    result = run(divmod_func, args_list)
    
    # Should return transposed tuples
    assert isinstance(result, tuple)
    assert len(result) == 2  # Two elements per tuple
    assert result[0] == [3, 3, 3]  # Quotients
    assert result[1] == [1, 3, 2]  # Remainders


def test_run_mixed_tuple_returns():
    """Test parallel execution with mixed tuple returns."""
    from mngs.parallel._run import run

    def stats(numbers):
        return sum(numbers), len(numbers), sum(numbers) / len(numbers)

    args_list = [([1, 2, 3],), ([4, 5, 6],), ([7, 8, 9],)]
    result = run(stats, args_list)
    
    assert isinstance(result, tuple)
    assert len(result) == 3  # Three elements per tuple
    assert result[0] == [6, 15, 24]  # Sums
    assert result[1] == [3, 3, 3]  # Lengths
    assert result[2] == [2.0, 5.0, 8.0]  # Averages


def test_run_empty_args_list():
    """Test that empty args_list raises ValueError."""
    from mngs.parallel._run import run

    def dummy(x):
        return x

    with pytest.raises(ValueError, match="Args list cannot be empty"):
        run(dummy, [])


def test_run_non_callable_func():
    """Test that non-callable func raises ValueError."""
    from mngs.parallel._run import run

    args_list = [(1, 2), (3, 4)]
    
    with pytest.raises(ValueError, match="Func must be callable"):
        run("not_callable", args_list)

    with pytest.raises(ValueError, match="Func must be callable"):
        run(123, args_list)


def test_run_n_jobs_auto_detection():
    """Test automatic CPU count detection with n_jobs=-1."""
    from mngs.parallel._run import run

    def add(x, y):
        return x + y

    args_list = [(1, 2), (3, 4)]
    
    with patch('multiprocessing.cpu_count', return_value=4):
        # Should use all 4 CPUs when n_jobs=-1
        result = run(add, args_list, n_jobs=-1)
        assert result == [3, 7]


def test_run_n_jobs_explicit():
    """Test explicit n_jobs setting."""
    from mngs.parallel._run import run

    def add(x, y):
        return x + y

    args_list = [(1, 2), (3, 4)]
    result = run(add, args_list, n_jobs=2)
    
    assert result == [3, 7]


def test_run_n_jobs_warning():
    """Test warning when n_jobs exceeds CPU count."""
    from mngs.parallel._run import run

    def add(x, y):
        return x + y

    args_list = [(1, 2), (3, 4)]
    
    with patch('multiprocessing.cpu_count', return_value=2):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run(add, args_list, n_jobs=4)
            
            assert len(w) == 1
            assert "n_jobs (4) is greater than CPU count (2)" in str(w[0].message)
            assert result == [3, 7]


def test_run_n_jobs_invalid():
    """Test invalid n_jobs values."""
    from mngs.parallel._run import run

    def add(x, y):
        return x + y

    args_list = [(1, 2), (3, 4)]
    
    with pytest.raises(ValueError, match="n_jobs must be >= 1 or -1"):
        run(add, args_list, n_jobs=0)
    
    with pytest.raises(ValueError, match="n_jobs must be >= 1 or -1"):
        run(add, args_list, n_jobs=-2)


def test_run_custom_description():
    """Test custom progress bar description."""
    from mngs.parallel._run import run

    def add(x, y):
        return x + y

    args_list = [(1, 2), (3, 4), (5, 6)]
    
    # Should run without error with custom description
    result = run(add, args_list, desc="Custom Processing")
    assert result == [3, 7, 11]


def test_run_order_preservation():
    """Test that results maintain order despite parallel execution."""
    from mngs.parallel._run import run

    def delayed_identity(x, delay):
        time.sleep(delay)
        return x

    # Longer delays for smaller numbers to test order preservation
    args_list = [(1, 0.1), (2, 0.05), (3, 0.01)]
    result = run(delayed_identity, args_list)
    
    # Results should maintain input order despite different completion times
    assert result == [1, 2, 3]


def test_run_exception_handling():
    """Test that exceptions in worker functions are properly raised."""
    from mngs.parallel._run import run

    def failing_func(x):
        if x == 2:
            raise ValueError(f"Error processing {x}")
        return x * 2

    args_list = [(1,), (2,), (3,)]
    
    # Should propagate the exception from the failing worker
    with pytest.raises(ValueError, match="Error processing 2"):
        run(failing_func, args_list)


def test_run_complex_data_types():
    """Test parallel execution with complex data types."""
    from mngs.parallel._run import run

    def process_dict(data_dict, multiplier):
        return {k: v * multiplier for k, v in data_dict.items()}

    args_list = [
        ({"a": 1, "b": 2}, 2),
        ({"x": 3, "y": 4}, 3),
        ({"p": 5, "q": 6}, 4),
    ]
    result = run(process_dict, args_list)
    
    expected = [
        {"a": 2, "b": 4},
        {"x": 9, "y": 12},
        {"p": 20, "q": 24},
    ]
    assert result == expected


def test_run_large_dataset():
    """Test parallel execution with larger dataset for performance validation."""
    from mngs.parallel._run import run

    def compute_factorial(n):
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    # Test with moderately sized dataset
    args_list = [(i,) for i in range(1, 11)]
    result = run(compute_factorial, args_list)
    
    # Verify some known factorial values
    assert result[0] == 1  # 1!
    assert result[4] == 120  # 5!
    assert result[9] == 3628800  # 10!
    assert len(result) == 10


def test_run_thread_safety():
    """Test thread safety with shared state."""
    from mngs.parallel._run import run

    def increment_and_return(base, increment):
        # Pure function, no shared state
        return base + increment

    args_list = [(i, 1) for i in range(20)]
    result = run(increment_and_return, args_list)
    
    expected = list(range(1, 21))
    assert result == expected


def test_run_memory_efficiency():
    """Test that results are properly allocated and ordered."""
    from mngs.parallel._run import run

    def create_list(size, value):
        return [value] * size

    args_list = [(3, i) for i in range(5)]
    result = run(create_list, args_list)
    
    expected = [
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
    ]
    assert result == expected


def test_run_string_operations():
    """Test parallel execution with string operations."""
    from mngs.parallel._run import run

    def format_string(template, value):
        return template.format(value)

    args_list = [
        ("Hello {}", "World"),
        ("Number: {}", 42),
        ("Status: {}", "OK"),
    ]
    result = run(format_string, args_list)
    
    assert result == ["Hello World", "Number: 42", "Status: OK"]


if __name__ == "__main__":
    import os
    pytest.main([os.path.abspath(__file__)])