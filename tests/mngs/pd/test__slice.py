#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 08:00:00 (ywatanabe)"
# File: ./tests/mngs/pd/test__slice.py

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
import builtins


class TestSliceBasic:
    """Test basic functionality of slice function."""

    def test_slice_by_indices(self):
        """Test slicing DataFrame by row indices using slice object."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})

        result = slice(df, builtins.slice(1, 4))

        assert len(result) == 3
        assert result["A"].tolist() == [2, 3, 4]
        assert result["B"].tolist() == ["b", "c", "d"]
        assert result.index.tolist() == [1, 2, 3]

    def test_slice_from_start(self):
        """Test slicing from start of DataFrame."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [10, 20, 30, 40], "B": [100, 200, 300, 400]})

        result = slice(df, builtins.slice(None, 2))

        assert len(result) == 2
        assert result["A"].tolist() == [10, 20]
        assert result["B"].tolist() == [100, 200]

    def test_slice_to_end(self):
        """Test slicing to end of DataFrame."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})

        result = slice(df, builtins.slice(3, None))

        assert len(result) == 2
        assert result["A"].tolist() == [4, 5]
        assert result["B"].tolist() == [40, 50]

    def test_slice_with_step(self):
        """Test slicing with step parameter."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": list(range(10)), "B": list(range(10, 20))})

        result = slice(df, builtins.slice(0, 10, 2))

        assert len(result) == 5
        assert result["A"].tolist() == [0, 2, 4, 6, 8]
        assert result["B"].tolist() == [10, 12, 14, 16, 18]


class TestSliceByConditions:
    """Test slicing by conditions using dictionary."""

    def test_single_condition(self):
        """Test slicing with single condition."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2, 3, 2, 1], "B": ["x", "y", "z", "y", "x"]})

        result = slice(df, {"A": 2})

        assert len(result) == 2
        assert result["A"].tolist() == [2, 2]
        assert result["B"].tolist() == ["y", "y"]
        assert result.index.tolist() == [1, 3]

    def test_multiple_conditions(self):
        """Test slicing with multiple conditions."""
        from mngs.pd._slice import slice

        df = pd.DataFrame(
            {
                "A": [1, 1, 2, 2, 3],
                "B": ["x", "y", "x", "y", "x"],
                "C": [10, 20, 30, 40, 50],
            }
        )

        result = slice(df, {"A": 2, "B": "x"})

        assert len(result) == 1
        assert result["A"].tolist() == [2]
        assert result["B"].tolist() == ["x"]
        assert result["C"].tolist() == [30]

    def test_list_condition(self):
        """Test slicing with list values in conditions."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})

        result = slice(df, {"A": [2, 4, 5]})

        assert len(result) == 3
        assert result["A"].tolist() == [2, 4, 5]
        assert result["B"].tolist() == ["b", "d", "e"]


class TestColumnSlicing:
    """Test column selection functionality."""

    def test_select_single_column(self):
        """Test selecting single column."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

        result = slice(df, columns=["B"])

        assert list(result.columns) == ["B"]
        assert result["B"].tolist() == [4, 5, 6]

    def test_select_multiple_columns(self):
        """Test selecting multiple columns."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]})

        result = slice(df, columns=["A", "C", "D"])

        assert list(result.columns) == ["A", "C", "D"]
        assert "B" not in result.columns

    def test_reorder_columns(self):
        """Test that column order follows the specified list."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})

        result = slice(df, columns=["C", "A", "B"])

        assert list(result.columns) == ["C", "A", "B"]


class TestCombinedSlicing:
    """Test combining row and column slicing."""

    def test_slice_rows_and_columns(self):
        """Test slicing both rows and columns."""
        from mngs.pd._slice import slice

        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": ["a", "b", "c", "d", "e"],
                "D": [100, 200, 300, 400, 500],
            }
        )

        result = slice(df, builtins.slice(1, 4), columns=["B", "C"])

        assert len(result) == 3
        assert list(result.columns) == ["B", "C"]
        assert result["B"].tolist() == [20, 30, 40]
        assert result["C"].tolist() == ["b", "c", "d"]

    def test_conditions_and_columns(self):
        """Test using conditions and column selection together."""
        from mngs.pd._slice import slice

        df = pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],
                "value": [10, 20, 30, 40, 50],
                "extra1": [1, 2, 3, 4, 5],
                "extra2": [6, 7, 8, 9, 10],
            }
        )

        result = slice(df, {"category": "A"}, columns=["category", "value"])

        assert len(result) == 3
        assert list(result.columns) == ["category", "value"]
        assert result["value"].tolist() == [10, 30, 50]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test slicing empty DataFrame."""
        from mngs.pd._slice import slice

        df = pd.DataFrame()
        result = slice(df, builtins.slice(0, 10))

        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_no_conditions(self):
        """Test with no slicing conditions."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = slice(df)

        pd.testing.assert_frame_equal(result, df)

    def test_no_matching_conditions(self):
        """Test when conditions match no rows."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})

        result = slice(df, {"A": 999})

        assert len(result) == 0
        assert list(result.columns) == ["A", "B"]

    def test_out_of_bounds_slice(self):
        """Test slice indices beyond DataFrame bounds."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2, 3]})

        # Slice beyond bounds should work without error
        result = slice(df, builtins.slice(10, 20))
        assert len(result) == 0

        result = slice(df, builtins.slice(-10, -5))
        assert len(result) == 0

    def test_negative_slice_indices(self):
        """Test negative indices in slice."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "c", "d", "e"]})

        result = slice(df, builtins.slice(-3, -1))

        assert len(result) == 2
        assert result["A"].tolist() == [3, 4]
        assert result["B"].tolist() == ["c", "d"]


class TestDataTypes:
    """Test with various data types."""

    def test_mixed_dtypes(self):
        """Test slicing DataFrame with mixed data types."""
        from mngs.pd._slice import slice

        df = pd.DataFrame(
            {
                "int": [1, 2, 3, 4],
                "float": [1.1, 2.2, 3.3, 4.4],
                "str": ["a", "b", "c", "d"],
                "bool": [True, False, True, False],
                "date": pd.date_range("2023-01-01", periods=4),
            }
        )

        result = slice(df, builtins.slice(1, 3))

        assert len(result) == 2
        assert result["int"].tolist() == [2, 3]
        assert result["float"].tolist() == [2.2, 3.3]
        assert result["bool"].tolist() == [False, True]

    def test_nan_values(self):
        """Test slicing with NaN values."""
        from mngs.pd._slice import slice

        df = pd.DataFrame(
            {"A": [1, np.nan, 3, np.nan, 5], "B": ["a", "b", np.nan, "d", "e"]}
        )

        # Slice should preserve NaN values
        result = slice(df, builtins.slice(1, 4))

        assert len(result) == 3
        assert pd.isna(result["A"].iloc[0])
        assert result["A"].iloc[1] == 3
        assert pd.isna(result["A"].iloc[2])


class TestIndexPreservation:
    """Test DataFrame index handling."""

    def test_custom_index_preservation(self):
        """Test that custom index is preserved."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2, 3, 4]}, index=["w", "x", "y", "z"])

        result = slice(df, builtins.slice(1, 3))

        assert list(result.index) == ["x", "y"]
        assert result.loc["x", "A"] == 2
        assert result.loc["y", "A"] == 3

    def test_multiindex(self):
        """Test slicing with MultiIndex."""
        from mngs.pd._slice import slice

        arrays = [["A", "A", "B", "B"], [1, 2, 1, 2]]
        index = pd.MultiIndex.from_arrays(arrays)
        df = pd.DataFrame({"value": [10, 20, 30, 40]}, index=index)

        result = slice(df, builtins.slice(1, 3))

        assert len(result) == 2
        assert result["value"].tolist() == [20, 30]


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_data_filtering_workflow(self):
        """Test typical data filtering workflow."""
        from mngs.pd._slice import slice

        # Sample sales data
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10),
                "product": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
                "quantity": [10, 20, 15, 5, 25, 30, 10, 35, 20, 15],
                "revenue": [100, 400, 150, 75, 500, 300, 150, 700, 200, 225],
            }
        )

        # Filter for product A with revenue > 100
        result = slice(df, {"product": "A"}, columns=["date", "product", "revenue"])
        result = result[result["revenue"] > 100]

        assert len(result) == 3
        assert all(result["product"] == "A")
        assert all(result["revenue"] > 100)

    def test_time_series_window(self):
        """Test extracting time series window."""
        from mngs.pd._slice import slice

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="H"),
                "value": np.random.randn(100),
            }
        )

        # Get specific time window
        result = slice(df, builtins.slice(24, 48))  # Hours 24-47

        assert len(result) == 24
        assert result["timestamp"].iloc[0].hour == 0  # Next day start
        assert result["timestamp"].iloc[0].day == 2


class TestDocstringExamples:
    """Test examples from the docstring."""

    def test_docstring_slice_example(self):
        """Test slice example from docstring."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "x"]})

        # Slice by row indices
        result = slice(df, builtins.slice(0, 2))
        assert len(result) == 2
        assert result["A"].tolist() == [1, 2]
        assert result["B"].tolist() == ["x", "y"]

    def test_docstring_conditions_example(self):
        """Test conditions example from docstring."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "x"]})

        # Slice by conditions
        result = slice(df, {"A": [1, 2], "B": "x"})
        assert len(result) == 1
        assert result["A"].tolist() == [1]
        assert result["B"].tolist() == ["x"]

    def test_docstring_columns_example(self):
        """Test columns example from docstring."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "x"]})

        # Slice columns
        result = slice(df, columns=["A"])
        assert list(result.columns) == ["A"]
        assert len(result) == 3  # All rows preserved


class TestCopyBehavior:
    """Test that slice returns a copy, not a view."""

    def test_returns_copy(self):
        """Test that modifications to result don't affect original."""
        from mngs.pd._slice import slice

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        result = slice(df, builtins.slice(0, 2))
        result["A"] = [99, 98]

        # Original should be unchanged
        assert df["A"].tolist() == [1, 2, 3]
        assert result["A"].tolist() == [99, 98]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
