import pytest
import numpy as np
import pandas as pd
import mngs


class TestP2StarsBasic:
    """Test basic p2stars functionality."""

    def test_import(self):
        """Test function can be imported."""
        assert hasattr(mngs.stats, "p2stars")

    def test_float_input(self):
        """Test p2stars with float inputs."""
        assert mngs.stats.p2stars(0.0005) == "***"
        assert mngs.stats.p2stars(0.001) == "***"
        assert mngs.stats.p2stars(0.005) == "**"
        assert mngs.stats.p2stars(0.01) == "**"
        assert mngs.stats.p2stars(0.03) == "*"
        assert mngs.stats.p2stars(0.05) == "*"
        assert mngs.stats.p2stars(0.06) == ""
        assert mngs.stats.p2stars(0.1) == ""
        assert mngs.stats.p2stars(0.5) == ""

    def test_string_input(self):
        """Test p2stars with string inputs."""
        assert mngs.stats.p2stars("0.0005") == "***"
        assert mngs.stats.p2stars("0.03") == "*"
        assert mngs.stats.p2stars("0.1") == ""
        assert mngs.stats.p2stars("1e-4") == "***"
        assert mngs.stats.p2stars("2e-3") == "**"

    def test_int_input(self):
        """Test p2stars with integer inputs (0 or 1)."""
        assert mngs.stats.p2stars(0) == "***"
        assert mngs.stats.p2stars(1) == ""

    def test_ns_parameter(self):
        """Test the ns parameter for non-significant values."""
        # With ns=False (default)
        assert mngs.stats.p2stars(0.1, ns=False) == ""
        assert mngs.stats.p2stars(0.5, ns=False) == ""
        
        # With ns=True
        assert mngs.stats.p2stars(0.1, ns=True) == "ns"
        assert mngs.stats.p2stars(0.5, ns=True) == "ns"
        
        # Significant values should not be affected
        assert mngs.stats.p2stars(0.01, ns=True) == "**"
        assert mngs.stats.p2stars(0.05, ns=True) == "*"

    def test_boundary_values(self):
        """Test boundary p-values."""
        # Exact boundaries
        assert mngs.stats.p2stars(0.001) == "***"
        assert mngs.stats.p2stars(0.01) == "**"
        assert mngs.stats.p2stars(0.05) == "*"
        
        # Just above boundaries
        assert mngs.stats.p2stars(0.0011) == "**"
        assert mngs.stats.p2stars(0.011) == "*"
        assert mngs.stats.p2stars(0.051) == ""

    def test_missing_values(self):
        """Test handling of missing values."""
        assert mngs.stats.p2stars("NA") == "NA"
        assert mngs.stats.p2stars("na") == "NA"
        assert mngs.stats.p2stars("nan") == "NA"
        assert mngs.stats.p2stars("null") == "NA"
        assert mngs.stats.p2stars("") == "NA"

    def test_invalid_values(self):
        """Test error handling for invalid values."""
        # Negative values
        with pytest.raises(ValueError, match="P-value must be between 0 and 1"):
            mngs.stats.p2stars(-0.1)
        
        # Values > 1
        with pytest.raises(ValueError, match="P-value must be between 0 and 1"):
            mngs.stats.p2stars(1.5)
        
        # Invalid strings
        with pytest.raises(ValueError, match="Invalid p-value"):
            mngs.stats.p2stars("abc")


class TestP2StarsDataFrame:
    """Test p2stars with DataFrame inputs."""

    def test_basic_dataframe(self):
        """Test p2stars with a basic DataFrame."""
        df = pd.DataFrame({
            'p_value': [0.001, 0.03, 0.1, 0.5]
        })
        
        result = mngs.stats.p2stars(df)
        
        assert 'p_value_stars' in result.columns
        assert result['p_value_stars'].tolist() == ['***', '*', '', '']

    def test_dataframe_with_multiple_p_columns(self):
        """Test DataFrame with multiple p-value columns."""
        df = pd.DataFrame({
            'group': ['A', 'B', 'C'],
            'p_value': [0.001, 0.03, 0.1],
            'pval_adjusted': [0.005, 0.06, 0.2],
            'p-val': [0.0001, 0.01, 0.05]
        })
        
        result = mngs.stats.p2stars(df)
        
        # Should have stars columns for each p-value column
        assert 'p_value_stars' in result.columns
        assert 'pval_adjusted_stars' in result.columns
        assert 'p-val_stars' in result.columns
        
        # Check values
        assert result['p_value_stars'].tolist() == ['***', '*', '']
        assert result['pval_adjusted_stars'].tolist() == ['**', '', '']
        assert result['p-val_stars'].tolist() == ['***', '**', '*']

    def test_dataframe_column_order(self):
        """Test that stars columns are inserted after their p-value columns."""
        df = pd.DataFrame({
            'group': ['A', 'B'],
            'p_value': [0.01, 0.1],
            'effect_size': [0.5, 0.3]
        })
        
        result = mngs.stats.p2stars(df)
        
        # Check column order
        cols = result.columns.tolist()
        p_idx = cols.index('p_value')
        stars_idx = cols.index('p_value_stars')
        
        assert stars_idx == p_idx + 1

    def test_dataframe_with_na_values(self):
        """Test DataFrame with NA values."""
        df = pd.DataFrame({
            'p_value': [0.001, 0.03, np.nan, 'NA', None]
        })
        
        result = mngs.stats.p2stars(df)
        
        # Check handling of different NA representations
        assert result['p_value_stars'].iloc[0] == '***'
        assert result['p_value_stars'].iloc[1] == '*'
        # NaN and None might raise errors or return specific values

    def test_dataframe_with_ns_parameter(self):
        """Test DataFrame processing with ns parameter."""
        df = pd.DataFrame({
            'p_value': [0.001, 0.05, 0.1, 0.5]
        })
        
        result = mngs.stats.p2stars(df, ns=True)
        
        assert result['p_value_stars'].tolist() == ['***', '*', 'ns', 'ns']

    def test_no_p_value_columns(self):
        """Test DataFrame with no p-value columns."""
        df = pd.DataFrame({
            'group': ['A', 'B'],
            'value': [1, 2]
        })
        
        # Should raise assertion error
        with pytest.raises(AssertionError, match="No p-value columns found"):
            mngs.stats.p2stars(df)

    def test_mixed_types_in_column(self):
        """Test DataFrame with mixed types in p-value column."""
        df = pd.DataFrame({
            'p_value': [0.001, "0.03", 0.1, "NA"]
        })
        
        result = mngs.stats.p2stars(df)
        
        assert result['p_value_stars'].tolist() == ['***', '*', '', 'NA']


class TestP2StarsEdgeCases:
    """Test edge cases and special scenarios."""

    def test_scientific_notation(self):
        """Test handling of scientific notation."""
        assert mngs.stats.p2stars("1e-5") == "***"
        assert mngs.stats.p2stars("5e-3") == "**"
        assert mngs.stats.p2stars("3e-2") == "*"
        assert mngs.stats.p2stars("1e-1") == ""

    def test_very_small_p_values(self):
        """Test very small p-values."""
        assert mngs.stats.p2stars(1e-10) == "***"
        assert mngs.stats.p2stars(1e-100) == "***"
        assert mngs.stats.p2stars(0.0) == "***"

    def test_whitespace_handling(self):
        """Test string inputs with whitespace."""
        assert mngs.stats.p2stars(" 0.01 ") == "**"
        assert mngs.stats.p2stars("\t0.05\n") == "*"
        assert mngs.stats.p2stars("  NA  ") == "NA"

    def test_case_insensitive_na(self):
        """Test case-insensitive NA handling."""
        assert mngs.stats.p2stars("NA") == "NA"
        assert mngs.stats.p2stars("na") == "NA"
        assert mngs.stats.p2stars("Na") == "NA"
        assert mngs.stats.p2stars("nA") == "NA"

    def test_invalid_input_types(self):
        """Test error handling for invalid input types."""
        with pytest.raises(ValueError, match="Input must be a float, string, or a pandas DataFrame"):
            mngs.stats.p2stars([0.01, 0.05])
        
        with pytest.raises(ValueError, match="Input must be a float, string, or a pandas DataFrame"):
            mngs.stats.p2stars(np.array([0.01, 0.05]))


class TestP2StarsIntegration:
    """Test integration with other stats functions."""

    def test_with_correlation_results(self):
        """Test p2stars with correlation test results."""
        # Simulate correlation test results
        df = pd.DataFrame({
            'variable1': ['x', 'x', 'y'],
            'variable2': ['y', 'z', 'z'],
            'correlation': [0.8, 0.3, -0.5],
            'p_value': [0.001, 0.05, 0.1]
        })
        
        result = mngs.stats.p2stars(df)
        
        assert 'p_value_stars' in result.columns
        assert result['p_value_stars'].tolist() == ['***', '*', '']

    def test_with_multiple_testing_correction(self):
        """Test p2stars after multiple testing correction."""
        # Original p-values
        p_values = [0.001, 0.01, 0.03, 0.04, 0.05]
        
        df = pd.DataFrame({
            'test': ['test1', 'test2', 'test3', 'test4', 'test5'],
            'p_value': p_values,
            'p_value_bonf': [p * 5 for p in p_values]  # Bonferroni correction
        })
        
        result = mngs.stats.p2stars(df)
        
        # Original p-values stars
        assert result['p_value_stars'].tolist() == ['***', '**', '*', '*', '*']
        
        # Corrected p-values stars (more conservative)
        assert result['p_value_bonf_stars'].tolist() == ['**', '*', '', '', '']


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/stats/_p2stars.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-06 10:39:57 (ywatanabe)"
#
# import pandas as pd
# import re
# from typing import Union, List
#
#
# def p2stars(
#     input_data: Union[float, str, pd.DataFrame], ns: bool = False
# ) -> Union[str, pd.DataFrame]:
#     """
#     Convert p-value(s) to significance stars.
#
#     Example
#     -------
#     >>> p2stars(0.0005)
#     '***'
#     >>> p2stars("0.03")
#     '*'
#     >>> p2stars("1e-4")
#     '***'
#     >>> df = pd.DataFrame({'p_value': [0.001, "0.03", 0.1, "NA"]})
#     >>> p2stars(df)
#        p_value
#     0    0.001 ***
#     1    0.030   *
#     2    0.100
#     3       NA  NA
#
#     Parameters
#     ----------
#     input_data : float, str, or pd.DataFrame
#         The p-value or DataFrame containing p-values to convert.
#         For DataFrame, columns matching re.search(r'p[_.-]?val', col.lower()) are considered.
#     ns : bool, optional
#         Whether to return 'n.s.' for non-significant results (default is False)
#
#     Returns
#     -------
#     str or pd.DataFrame
#         Significance stars or DataFrame with added stars column
#     """
#     if isinstance(input_data, (float, int, str)):
#         return _p2stars_str(input_data, ns)
#     elif isinstance(input_data, pd.DataFrame):
#         return _p2stars_pd(input_data, ns)
#     else:
#         raise ValueError("Input must be a float, string, or a pandas DataFrame")
# ... (truncated for brevity)

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/stats/_p2stars.py
# --------------------------------------------------------------------------------