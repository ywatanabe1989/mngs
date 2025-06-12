#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-05-31 22:20:00 (claude)"
# File: ./tests/mngs/gen/test__to_odd.py

"""
Comprehensive tests for mngs.gen._to_odd module.

This module tests:
- to_odd function with various numeric inputs
- Edge cases and special values
- Type handling
"""

import pytest
import numpy as np


class TestToOddBasic:
    """Test basic to_odd functionality."""

    def test_even_integers(self):
        """Test conversion of even integers to odd."""
        from mngs.gen._to_odd import to_odd

        assert to_odd(2) == 1
        assert to_odd(4) == 3
        assert to_odd(6) == 5
        assert to_odd(8) == 7
        assert to_odd(10) == 9
        assert to_odd(100) == 99

    def test_odd_integers(self):
        """Test that odd integers remain unchanged."""
        from mngs.gen._to_odd import to_odd

        assert to_odd(1) == 1
        assert to_odd(3) == 3
        assert to_odd(5) == 5
        assert to_odd(7) == 7
        assert to_odd(9) == 9
        assert to_odd(99) == 99

    def test_zero(self):
        """Test conversion of zero."""
        from mngs.gen._to_odd import to_odd

        # 0 is even, so should become -1
        assert to_odd(0) == -1

    def test_negative_even(self):
        """Test conversion of negative even integers."""
        from mngs.gen._to_odd import to_odd

        assert to_odd(-2) == -3
        assert to_odd(-4) == -5
        assert to_odd(-6) == -7
        assert to_odd(-8) == -9

    def test_negative_odd(self):
        """Test that negative odd integers remain unchanged."""
        from mngs.gen._to_odd import to_odd

        assert to_odd(-1) == -1
        assert to_odd(-3) == -3
        assert to_odd(-5) == -5
        assert to_odd(-7) == -7
        assert to_odd(-9) == -9


class TestToOddFloats:
    """Test to_odd with floating point numbers."""

    def test_float_truncation(self):
        """Test that floats are truncated, not rounded."""
        from mngs.gen._to_odd import to_odd

        # Positive floats
        assert to_odd(5.1) == 5
        assert to_odd(5.5) == 5
        assert to_odd(5.9) == 5
        assert to_odd(6.1) == 5
        assert to_odd(6.5) == 5
        assert to_odd(6.9) == 5

    def test_float_even_base(self):
        """Test floats with even integer part."""
        from mngs.gen._to_odd import to_odd

        assert to_odd(4.1) == 3
        assert to_odd(4.5) == 3
        assert to_odd(4.9) == 3
        assert to_odd(8.3) == 7

    def test_negative_floats(self):
        """Test negative floats."""
        from mngs.gen._to_odd import to_odd

        assert to_odd(-5.1) == -5
        assert to_odd(-5.5) == -5
        assert to_odd(-5.9) == -5
        assert to_odd(-6.1) == -7
        assert to_odd(-6.5) == -7
        assert to_odd(-6.9) == -7

    def test_documentation_examples(self):
        """Test examples from the docstring."""
        from mngs.gen._to_odd import to_odd

        assert to_odd(6) == 5
        assert to_odd(7) == 7
        assert to_odd(5.8) == 5


class TestToOddEdgeCases:
    """Test edge cases for to_odd function."""

    def test_large_numbers(self):
        """Test with very large numbers."""
        from mngs.gen._to_odd import to_odd

        assert to_odd(1000000) == 999999
        assert to_odd(1000001) == 1000001
        assert to_odd(10**9) == 10**9 - 1
        assert to_odd(10**9 + 1) == 10**9 + 1

    def test_special_floats(self):
        """Test with special float values."""
        from mngs.gen._to_odd import to_odd

        # Very small positive numbers should become -1
        assert to_odd(0.1) == -1
        assert to_odd(0.5) == -1
        assert to_odd(0.9) == -1

        # Very small negative numbers
        assert to_odd(-0.1) == -1
        assert to_odd(-0.5) == -1
        assert to_odd(-0.9) == -1

    def test_numpy_types(self):
        """Test with NumPy numeric types."""
        from mngs.gen._to_odd import to_odd

        # NumPy integers
        assert to_odd(np.int32(6)) == 5
        assert to_odd(np.int64(7)) == 7
        assert to_odd(np.int16(8)) == 7

        # NumPy floats
        assert to_odd(np.float32(6.5)) == 5
        assert to_odd(np.float64(7.9)) == 7

    def test_boolean_inputs(self):
        """Test with boolean inputs (which convert to 0/1)."""
        from mngs.gen._to_odd import to_odd

        assert to_odd(True) == 1  # True -> 1 (already odd)
        assert to_odd(False) == -1  # False -> 0 -> -1


class TestToOddConsecutive:
    """Test to_odd with consecutive inputs."""

    def test_consecutive_integers(self):
        """Test pattern with consecutive integers."""
        from mngs.gen._to_odd import to_odd

        results = [to_odd(i) for i in range(10)]
        expected = [-1, 1, 1, 3, 3, 5, 5, 7, 7, 9]
        assert results == expected

    def test_consecutive_negative(self):
        """Test pattern with consecutive negative integers."""
        from mngs.gen._to_odd import to_odd

        results = [to_odd(i) for i in range(-5, 6)]
        expected = [-5, -5, -3, -3, -1, -1, 1, 1, 3, 3, 5]
        assert results == expected

    def test_sequence_properties(self):
        """Test mathematical properties of the conversion."""
        from mngs.gen._to_odd import to_odd

        # Property: result is always odd
        for n in range(-20, 21):
            result = to_odd(n)
            assert result % 2 != 0, f"to_odd({n})={result} is not odd"

        # Property: result <= input
        for n in range(-20, 21):
            result = to_odd(n)
            assert result <= n, f"to_odd({n})={result} is greater than input"

        # Property: difference is at most 1
        for n in range(-20, 21):
            result = to_odd(n)
            assert n - result <= 1, f"to_odd({n})={result} differs by more than 1"


class TestToOddParameterized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            (0, -1),
            (1, 1),
            (2, 1),
            (3, 3),
            (4, 3),
            (5, 5),
            (-1, -1),
            (-2, -3),
            (-3, -3),
            (-4, -5),
            (10.7, 9),
            (-10.7, -11),
            (0.5, -1),
            (-0.5, -1),
        ],
    )
    def test_various_inputs(self, input_val, expected):
        """Test to_odd with various inputs using parametrization."""
        from mngs.gen._to_odd import to_odd

        assert to_odd(input_val) == expected

    @pytest.mark.parametrize("n", range(-100, 101, 10))
    def test_even_conversion_pattern(self, n):
        """Test that even numbers are converted correctly."""
        from mngs.gen._to_odd import to_odd

        if n % 2 == 0:
            # Even numbers should become n-1
            assert to_odd(n) == n - 1
        else:
            # Odd numbers should stay the same
            assert to_odd(n) == n


class TestToOddTypeHandling:
    """Test type handling and conversions."""

    def test_string_numeric(self):
        """Test behavior with numeric strings (should raise error)."""
        from mngs.gen._to_odd import to_odd

        # String inputs should cause TypeError in int()
        with pytest.raises(TypeError):
            to_odd("5")

        with pytest.raises(TypeError):
            to_odd("5.5")

    def test_none_input(self):
        """Test behavior with None input."""
        from mngs.gen._to_odd import to_odd

        with pytest.raises(TypeError):
            to_odd(None)

    def test_complex_numbers(self):
        """Test behavior with complex numbers."""
        from mngs.gen._to_odd import to_odd

        # Complex numbers don't support int() conversion
        with pytest.raises(TypeError):
            to_odd(3 + 4j)

    def test_infinity(self):
        """Test behavior with infinity."""
        from mngs.gen._to_odd import to_odd

        # Infinity can't be converted to int
        with pytest.raises((ValueError, OverflowError)):
            to_odd(float("inf"))

        with pytest.raises((ValueError, OverflowError)):
            to_odd(float("-inf"))

    def test_nan(self):
        """Test behavior with NaN."""
        from mngs.gen._to_odd import to_odd

        # NaN can't be converted to int
        with pytest.raises(ValueError):
            to_odd(float("nan"))


class TestToOddIntegration:
    """Integration tests for to_odd function."""

    def test_with_array_processing(self):
        """Test using to_odd with array processing."""
        from mngs.gen._to_odd import to_odd

        # Process array of values
        inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        results = [to_odd(x) for x in inputs]
        expected = [1, 1, 3, 3, 5, 5, 7, 7, 9, 9]
        assert results == expected

    def test_with_numpy_vectorize(self):
        """Test using to_odd with numpy vectorize."""
        from mngs.gen._to_odd import to_odd

        # Vectorize the function
        vec_to_odd = np.vectorize(to_odd)

        # Test on array
        inputs = np.array([1, 2, 3, 4, 5, 6])
        results = vec_to_odd(inputs)
        expected = np.array([1, 1, 3, 3, 5, 5])

        assert np.array_equal(results, expected)

    def test_use_case_kernel_sizes(self):
        """Test realistic use case: ensuring odd kernel sizes."""
        from mngs.gen._to_odd import to_odd

        # Common in image processing where kernels need odd sizes
        kernel_sizes = [3, 4, 5, 6, 7, 8, 9]
        odd_sizes = [to_odd(k) for k in kernel_sizes]

        # All should be odd
        assert all(s % 2 != 0 for s in odd_sizes)
        # Should preserve odd sizes
        assert odd_sizes == [3, 3, 5, 5, 7, 7, 9]


if __name__ == "__main__":
    # Run specific test file
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
