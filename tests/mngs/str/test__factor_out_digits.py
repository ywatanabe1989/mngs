#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-05 12:01:00 (ywatanabe)"
# File: ./tests/mngs/str/test__factor_out_digits.py

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch
from mngs.str._factor_out_digits import (
    factor_out_digits,
    auto_factor_axis,
    smart_tick_formatter,
    _format_factor_string,
    _factor_single_axis,
)


class TestFactorOutDigits:
    """Test cases for factor_out_digits function."""

    def test_basic_positive_values(self):
        """Test factoring with basic positive values."""
        values = [1000, 2000, 3000]
        factored, factor_str = factor_out_digits(values)
        
        assert factored == [1.0, 2.0, 3.0]
        assert factor_str == r"$\times 10^{3}$"

    def test_basic_negative_values(self):
        """Test factoring with negative values."""
        values = [-1000, -2000, -3000]
        factored, factor_str = factor_out_digits(values)
        
        assert factored == [-1.0, -2.0, -3.0]
        assert factor_str == r"$\times 10^{3}$"

    def test_small_decimal_values(self):
        """Test factoring with small decimal values."""
        values = [0.001, 0.002, 0.003]
        factored, factor_str = factor_out_digits(values)
        
        assert factored == [1.0, 2.0, 3.0]
        assert factor_str == r"$\times 10^{-3}$"

    def test_scientific_notation_input(self):
        """Test factoring with scientific notation input."""
        values = [1.5e6, 2.3e6, 4.1e6]
        factored, factor_str = factor_out_digits(values)
        
        expected = [1.5, 2.3, 4.1]
        np.testing.assert_array_almost_equal(factored, expected, decimal=1)
        assert factor_str == r"$\times 10^{6}$"

    def test_scalar_input(self):
        """Test factoring with scalar input."""
        value = 5000
        factored, factor_str = factor_out_digits(value)
        
        assert factored == 5.0
        assert factor_str == r"$\times 10^{3}$"

    def test_numpy_array_input(self):
        """Test factoring with numpy array input."""
        values = np.array([10000, 20000, 30000])
        factored, factor_str = factor_out_digits(values)
        
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(factored, expected)
        assert factor_str == r"$\times 10^{4}$"

    def test_mixed_magnitudes(self):
        """Test factoring with mixed order of magnitude values."""
        values = [100, 1000, 10000]
        factored, factor_str = factor_out_digits(values)
        
        # Should use average power (around 10^3)
        expected = [0.1, 1.0, 10.0]
        np.testing.assert_array_almost_equal(factored, expected, decimal=1)
        assert factor_str == r"$\times 10^{3}$"

    def test_precision_parameter(self):
        """Test precision parameter."""
        values = [1234.5678, 2345.6789]
        factored, factor_str = factor_out_digits(values, precision=1)
        
        expected = [1.2, 2.3]
        np.testing.assert_array_almost_equal(factored, expected, decimal=1)
        assert factor_str == r"$\times 10^{3}$"

    def test_min_factor_power_threshold(self):
        """Test min_factor_power threshold."""
        # Small power, should not be factored with default threshold
        values = [10, 20, 30]
        factored, factor_str = factor_out_digits(values, min_factor_power=3)
        
        assert factored == values  # Should return original values
        assert factor_str == ""

        # Same values with lower threshold
        factored, factor_str = factor_out_digits(values, min_factor_power=1)
        
        expected = [1.0, 2.0, 3.0]
        np.testing.assert_array_equal(factored, expected)
        assert factor_str == r"$\times 10^{1}$"

    def test_unicode_format(self):
        """Test Unicode superscript format."""
        values = [1000, 2000, 3000]
        factored, factor_str = factor_out_digits(values, return_latex=False, return_unicode=True)
        
        assert factored == [1.0, 2.0, 3.0]
        assert factor_str == "×10³"

    def test_plain_format(self):
        """Test plain format (no LaTeX, no Unicode)."""
        values = [1000, 2000, 3000]
        factored, factor_str = factor_out_digits(values, return_latex=False, return_unicode=False)
        
        assert factored == [1.0, 2.0, 3.0]
        assert factor_str == "×10^3"

    def test_zero_values(self):
        """Test handling of zero values."""
        values = [0, 0, 0]
        factored, factor_str = factor_out_digits(values)
        
        assert factored == values
        assert factor_str == ""

    def test_mixed_with_zeros(self):
        """Test handling of mixed values including zeros."""
        values = [0, 1000, 2000, 0, 3000]
        factored, factor_str = factor_out_digits(values)
        
        expected = [0.0, 1.0, 2.0, 0.0, 3.0]
        np.testing.assert_array_equal(factored, expected)
        assert factor_str == r"$\times 10^{3}$"

    def test_single_non_zero_value(self):
        """Test single non-zero value."""
        values = [5000]
        factored, factor_str = factor_out_digits(values)
        
        assert factored == [5.0]
        assert factor_str == r"$\times 10^{3}$"

    def test_negative_powers(self):
        """Test negative powers formatting."""
        values = [0.00001, 0.00002, 0.00003]
        factored, factor_str = factor_out_digits(values, min_factor_power=3)
        
        expected = [1.0, 2.0, 3.0]
        np.testing.assert_array_almost_equal(factored, expected, decimal=1)
        assert factor_str == r"$\times 10^{-5}$"

    def test_very_large_numbers(self):
        """Test very large numbers."""
        values = [1e12, 2e12, 3e12]
        factored, factor_str = factor_out_digits(values)
        
        expected = [1.0, 2.0, 3.0]
        np.testing.assert_array_almost_equal(factored, expected, decimal=1)
        assert factor_str == r"$\times 10^{12}$"


class TestFormatFactorString:
    """Test cases for _format_factor_string function."""

    def test_latex_format_positive(self):
        """Test LaTeX format with positive power."""
        result = _format_factor_string(3, latex=True)
        assert result == r"$\times 10^{3}$"

    def test_latex_format_negative(self):
        """Test LaTeX format with negative power."""
        result = _format_factor_string(-3, latex=True)
        assert result == r"$\times 10^{-3}$"

    def test_unicode_format_positive(self):
        """Test Unicode format with positive power."""
        result = _format_factor_string(3, latex=False, unicode_sup=True)
        assert result == "×10³"

    def test_unicode_format_negative(self):
        """Test Unicode format with negative power."""
        result = _format_factor_string(-3, latex=False, unicode_sup=True)
        assert result == "×10⁻³"

    def test_plain_format(self):
        """Test plain format."""
        result = _format_factor_string(3, latex=False, unicode_sup=False)
        assert result == "×10^3"

    def test_unicode_large_numbers(self):
        """Test Unicode format with multi-digit powers."""
        result = _format_factor_string(12, latex=False, unicode_sup=True)
        assert result == "×10¹²"

    def test_unicode_zero_power(self):
        """Test Unicode format with zero power."""
        result = _format_factor_string(0, latex=False, unicode_sup=True)
        assert result == "×10⁰"


class TestAutoFactorAxis:
    """Test cases for auto_factor_axis function."""

    def test_factor_x_axis(self):
        """Test factoring x-axis only."""
        fig, ax = plt.subplots()
        ax.plot([1000, 2000, 3000], [1, 2, 3])
        
        with patch.object(ax, 'set_xticks') as mock_set_x, \
             patch.object(ax, 'set_xticklabels') as mock_labels_x, \
             patch.object(ax, 'text') as mock_text:
            
            mock_ax = Mock()
            mock_ax.get_xticks.return_value = np.array([1000, 2000, 3000])
            
            auto_factor_axis(mock_ax, axis='x')
            
            # Should call x-axis methods
            mock_ax.get_xticks.assert_called_once()
        
        plt.close(fig)

    def test_factor_y_axis(self):
        """Test factoring y-axis only."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1000, 2000, 3000])
        
        with patch.object(ax, 'set_yticks') as mock_set_y, \
             patch.object(ax, 'set_yticklabels') as mock_labels_y, \
             patch.object(ax, 'text') as mock_text:
            
            mock_ax = Mock()
            mock_ax.get_yticks.return_value = np.array([1000, 2000, 3000])
            
            auto_factor_axis(mock_ax, axis='y')
            
            # Should call y-axis methods
            mock_ax.get_yticks.assert_called_once()
        
        plt.close(fig)

    def test_factor_both_axes(self):
        """Test factoring both axes."""
        fig, ax = plt.subplots()
        ax.plot([1000, 2000, 3000], [0.001, 0.002, 0.003])
        
        with patch('mngs.str._factor_out_digits._factor_single_axis') as mock_factor:
            auto_factor_axis(ax, axis='both')
            
            # Should call _factor_single_axis twice (for x and y)
            assert mock_factor.call_count == 2
        
        plt.close(fig)

    def test_custom_parameters(self):
        """Test auto_factor_axis with custom parameters."""
        fig, ax = plt.subplots()
        ax.plot([1000, 2000, 3000], [1, 2, 3])
        
        with patch('mngs.str._factor_out_digits._factor_single_axis') as mock_factor:
            auto_factor_axis(ax, precision=1, min_factor_power=2, 
                           return_latex=False, label_offset=(0.1, 0.9))
            
            # Verify parameters are passed correctly
            mock_factor.assert_called()
            args = mock_factor.call_args_list[0][0]  # First call args
            assert args[2] == 1  # precision
            assert args[3] == 2  # min_factor_power
        
        plt.close(fig)


class TestSmartTickFormatter:
    """Test cases for smart_tick_formatter function."""

    def test_basic_tick_formatting(self):
        """Test basic tick formatting."""
        values = [1000, 1500, 2000, 2500, 3000]
        positions, labels, factor_str = smart_tick_formatter(values)
        
        assert len(positions) <= 6  # max_ticks default
        assert len(labels) == len(positions)
        assert factor_str == r"$\times 10^{3}$"

    def test_without_factoring(self):
        """Test tick formatting without factoring."""
        values = [1000, 1500, 2000, 2500, 3000]
        positions, labels, factor_str = smart_tick_formatter(values, factor_out=False)
        
        assert factor_str == ""
        assert len(labels) == len(positions)

    def test_custom_max_ticks(self):
        """Test custom max_ticks parameter."""
        values = np.linspace(1000, 10000, 100)
        positions, labels, factor_str = smart_tick_formatter(values, max_ticks=3)
        
        assert len(positions) <= 3

    def test_small_values_no_factoring(self):
        """Test small values that shouldn't be factored."""
        values = [1, 2, 3, 4, 5]
        positions, labels, factor_str = smart_tick_formatter(values)
        
        assert factor_str == ""  # Should not factor small values

    def test_precision_formatting(self):
        """Test precision in tick label formatting."""
        values = [1234.567, 2345.678, 3456.789]
        positions, labels, factor_str = smart_tick_formatter(values, precision=1)
        
        # Labels should reflect precision
        for label in labels:
            if '.' in label:
                decimal_places = len(label.split('.')[1])
                assert decimal_places <= 1


class TestFactorSingleAxis:
    """Test cases for _factor_single_axis function."""

    def test_factor_x_axis_implementation(self):
        """Test _factor_single_axis for x-axis."""
        mock_ax = Mock()
        mock_ax.get_xticks.return_value = np.array([1000, 2000, 3000])
        mock_ax.transAxes = plt.gca().transAxes
        
        _factor_single_axis(mock_ax, 'x', 2, 3, True, False, (0.02, 0.98))
        
        # Verify x-axis methods were called
        mock_ax.get_xticks.assert_called_once()
        mock_ax.set_xticks.assert_called_once()
        mock_ax.set_xticklabels.assert_called_once()
        mock_ax.text.assert_called_once()

    def test_factor_y_axis_implementation(self):
        """Test _factor_single_axis for y-axis."""
        mock_ax = Mock()
        mock_ax.get_yticks.return_value = np.array([1000, 2000, 3000])
        mock_ax.transAxes = plt.gca().transAxes
        
        _factor_single_axis(mock_ax, 'y', 2, 3, True, False, (0.02, 0.98))
        
        # Verify y-axis methods were called
        mock_ax.get_yticks.assert_called_once()
        mock_ax.set_yticks.assert_called_once()
        mock_ax.set_yticklabels.assert_called_once()
        mock_ax.text.assert_called_once()

    def test_no_factoring_when_insufficient_power(self):
        """Test no factoring occurs when power is insufficient."""
        mock_ax = Mock()
        mock_ax.get_xticks.return_value = np.array([1, 2, 3])  # Small values
        
        _factor_single_axis(mock_ax, 'x', 2, 3, True, False, (0.02, 0.98))
        
        # Should not set new tick labels when no factoring occurs
        mock_ax.set_xticks.assert_not_called()
        mock_ax.set_xticklabels.assert_not_called()
        mock_ax.text.assert_not_called()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_list(self):
        """Test empty list input."""
        # Empty list returns original values and empty factor string
        factored, factor_str = factor_out_digits([])
        assert factored == []
        assert factor_str == ""

    def test_invalid_input_types(self):
        """Test invalid input types."""
        with pytest.raises((TypeError, AttributeError)):
            factor_out_digits("invalid")

    def test_nan_values(self):
        """Test handling of NaN values."""
        values = [np.nan, 1000, 2000]
        # Should handle gracefully or raise appropriate error
        try:
            factored, factor_str = factor_out_digits(values)
            # If it succeeds, verify result makes sense
            assert not np.isnan(factored[1])  # Non-NaN values should be processed
        except (ValueError, RuntimeWarning):
            # Acceptable to raise error for NaN inputs
            pass

    def test_infinite_values(self):
        """Test handling of infinite values."""
        values = [np.inf, 1000, 2000]
        # Function raises OverflowError when trying to convert inf to int
        with pytest.raises(OverflowError):
            factor_out_digits(values)

    def test_very_small_numbers_precision(self):
        """Test precision with very small numbers."""
        values = [1e-15, 2e-15, 3e-15]
        factored, factor_str = factor_out_digits(values, min_factor_power=10)
        
        # Should factor out if power is significant enough
        if factor_str:
            assert "10^{-15}" in factor_str or "10^{-14}" in factor_str

    def test_matplotlib_integration(self):
        """Test integration with matplotlib objects."""
        fig, ax = plt.subplots()
        x = np.array([1000, 2000, 3000])
        y = np.array([0.001, 0.002, 0.003])
        ax.plot(x, y)
        
        # Test that auto_factor_axis doesn't crash with real matplotlib objects
        try:
            auto_factor_axis(ax, axis='both')
            # If successful, verify the plot still renders
            assert ax.get_xlim() is not None
            assert ax.get_ylim() is not None
        except Exception as e:
            pytest.fail(f"auto_factor_axis failed with real matplotlib axes: {e}")
        finally:
            plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])


# EOF