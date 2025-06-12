#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-05-31 21:30:00 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/gen/test__list_packages.py

"""Tests for list_packages function."""

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mngs.gen._list_packages import list_packages, main


class MockDistribution:
    """Mock for importlib.metadata Distribution."""

    def __init__(self, name):
        self.name = name


class TestListPackages:
    """Test cases for list_packages function."""

    @patch("mngs.gen._list_packages.distributions")
    @patch("mngs.gen._list_packages.inspect_module")
    def test_basic_functionality(self, mock_inspect, mock_distributions):
        """Test basic package listing functionality."""
        # Setup mocks
        mock_distributions.return_value = [
            MockDistribution("numpy"),
            MockDistribution("pandas"),
            MockDistribution("scipy"),
        ]

        mock_inspect.return_value = pd.DataFrame(
            {"Name": ["numpy.array", "numpy.ndarray"]}
        )

        # Call function
        result = list_packages()

        # Verify
        assert isinstance(result, pd.DataFrame)
        assert "Name" in result.columns
        assert len(result) > 0
        assert mock_inspect.call_count == 3  # Called for each package

    @patch("mngs.gen._list_packages.distributions")
    @patch("mngs.gen._list_packages.inspect_module")
    def test_skip_patterns_filtering(self, mock_inspect, mock_distributions):
        """Test that problematic packages are skipped."""
        # Setup mocks with problematic packages
        mock_distributions.return_value = [
            MockDistribution("numpy"),
            MockDistribution("nvidia-cuda-runtime"),
            MockDistribution("pillow"),
            MockDistribution("pandas"),
        ]

        mock_inspect.return_value = pd.DataFrame({"Name": ["test.module"]})

        # Call function
        result = list_packages()

        # Verify only numpy and pandas were processed
        assert mock_inspect.call_count == 2
        called_packages = [call[0][0] for call in mock_inspect.call_args_list]
        assert "numpy" in called_packages
        assert "pandas" in called_packages
        assert "nvidia_cuda_runtime" not in called_packages
        assert "pillow" not in called_packages

    @patch("mngs.gen._list_packages.distributions")
    @patch("mngs.gen._list_packages.inspect_module")
    def test_safelist_prioritization(self, mock_inspect, mock_distributions):
        """Test that safelist packages are prioritized."""
        # Setup mocks
        mock_distributions.return_value = [
            MockDistribution("unknown-package"),
            MockDistribution("numpy"),
            MockDistribution("another-unknown"),
            MockDistribution("pandas"),
        ]

        mock_inspect.return_value = pd.DataFrame({"Name": ["test.module"]})

        # Call function
        list_packages()

        # Verify order - safelist packages should be processed first
        called_packages = [call[0][0] for call in mock_inspect.call_args_list]
        numpy_idx = called_packages.index("numpy")
        pandas_idx = called_packages.index("pandas")
        unknown_idx = called_packages.index("unknown_package")

        assert numpy_idx < unknown_idx
        assert pandas_idx < unknown_idx

    @patch("mngs.gen._list_packages.distributions")
    @patch("mngs.gen._list_packages.inspect_module")
    def test_error_handling_skip_errors_true(self, mock_inspect, mock_distributions):
        """Test error handling with skip_errors=True."""
        # Setup mocks
        mock_distributions.return_value = [
            MockDistribution("numpy"),
            MockDistribution("pandas"),
        ]

        # First call raises error, second succeeds
        mock_inspect.side_effect = [
            Exception("Import error"),
            pd.DataFrame({"Name": ["pandas.DataFrame"]}),
        ]

        # Call function with skip_errors=True
        result = list_packages(skip_errors=True)

        # Should continue and return pandas results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["Name"] == "pandas.DataFrame"

    @patch("mngs.gen._list_packages.distributions")
    @patch("mngs.gen._list_packages.inspect_module")
    def test_error_handling_skip_errors_false(self, mock_inspect, mock_distributions):
        """Test error handling with skip_errors=False."""
        # Setup mocks
        mock_distributions.return_value = [MockDistribution("numpy")]

        mock_inspect.side_effect = Exception("Import error")

        # Call function with skip_errors=False
        with pytest.raises(Exception, match="Import error"):
            list_packages(skip_errors=False)

    @patch("mngs.gen._list_packages.distributions")
    @patch("mngs.gen._list_packages.inspect_module")
    def test_empty_results(self, mock_inspect, mock_distributions):
        """Test handling of empty results."""
        # Setup mocks
        mock_distributions.return_value = [MockDistribution("numpy")]
        mock_inspect.return_value = pd.DataFrame()  # Empty dataframe

        # Call function
        result = list_packages()

        # Should return empty dataframe with correct columns
        assert isinstance(result, pd.DataFrame)
        assert "Name" in result.columns
        assert len(result) == 0

    @patch("mngs.gen._list_packages.distributions")
    @patch("mngs.gen._list_packages.inspect_module")
    def test_no_packages_found(self, mock_inspect, mock_distributions):
        """Test when no packages are found."""
        # Setup mocks
        mock_distributions.return_value = []

        # Call function
        result = list_packages()

        # Should return empty dataframe
        assert isinstance(result, pd.DataFrame)
        assert "Name" in result.columns
        assert len(result) == 0
        assert mock_inspect.call_count == 0

    @patch("mngs.gen._list_packages.distributions")
    @patch("mngs.gen._list_packages.inspect_module")
    def test_duplicate_removal(self, mock_inspect, mock_distributions):
        """Test that duplicates are removed from results."""
        # Setup mocks
        mock_distributions.return_value = [
            MockDistribution("numpy"),
            MockDistribution("pandas"),
        ]

        # Return dataframes with duplicates
        mock_inspect.side_effect = [
            pd.DataFrame({"Name": ["shared.module", "numpy.array"]}),
            pd.DataFrame({"Name": ["shared.module", "pandas.DataFrame"]}),
        ]

        # Call function
        result = list_packages()

        # Verify duplicates removed
        assert len(result) == 3  # Not 4
        assert result["Name"].tolist() == sorted(
            ["numpy.array", "pandas.DataFrame", "shared.module"]
        )

    @patch("mngs.gen._list_packages.distributions")
    @patch("mngs.gen._list_packages.inspect_module")
    def test_sorting(self, mock_inspect, mock_distributions):
        """Test that results are sorted by Name."""
        # Setup mocks
        mock_distributions.return_value = [MockDistribution("numpy")]

        mock_inspect.return_value = pd.DataFrame(
            {"Name": ["zzz.module", "aaa.module", "mmm.module"]}
        )

        # Call function
        result = list_packages()

        # Verify sorted
        assert result["Name"].tolist() == ["aaa.module", "mmm.module", "zzz.module"]

    @patch("mngs.gen._list_packages.distributions")
    @patch("mngs.gen._list_packages.inspect_module")
    def test_max_depth_parameter(self, mock_inspect, mock_distributions):
        """Test max_depth parameter is passed correctly."""
        # Setup mocks
        mock_distributions.return_value = [MockDistribution("numpy")]
        mock_inspect.return_value = pd.DataFrame({"Name": ["numpy.array"]})

        # Call function with max_depth
        list_packages(max_depth=3)

        # Verify max_depth was passed
        mock_inspect.assert_called_with(
            "numpy",
            docstring=False,
            print_output=False,
            columns=["Name"],
            root_only=True,
            max_depth=3,
            skip_depwarnings=True,
        )

    @patch("mngs.gen._list_packages.distributions")
    @patch("mngs.gen._list_packages.inspect_module")
    def test_root_only_parameter(self, mock_inspect, mock_distributions):
        """Test root_only parameter is passed correctly."""
        # Setup mocks
        mock_distributions.return_value = [MockDistribution("numpy")]
        mock_inspect.return_value = pd.DataFrame({"Name": ["numpy.array"]})

        # Call function with root_only=False
        list_packages(root_only=False)

        # Verify root_only was passed
        mock_inspect.assert_called_with(
            "numpy",
            docstring=False,
            print_output=False,
            columns=["Name"],
            root_only=False,
            max_depth=1,
            skip_depwarnings=True,
        )

    @patch("mngs.gen._list_packages.distributions")
    @patch("mngs.gen._list_packages.inspect_module")
    @patch("builtins.print")
    def test_verbose_output(self, mock_print, mock_inspect, mock_distributions):
        """Test verbose output for errors."""
        # Setup mocks
        mock_distributions.return_value = [MockDistribution("numpy")]
        mock_inspect.side_effect = Exception("Test error")

        # Call function with verbose=True
        result = list_packages(verbose=True, skip_errors=True)

        # Verify error was printed
        mock_print.assert_called_with("Error processing numpy: Test error")

    def test_recursion_limit_set(self):
        """Test that recursion limit is increased."""
        original_limit = sys.getrecursionlimit()

        with patch("mngs.gen._list_packages.distributions") as mock_dist:
            mock_dist.return_value = []
            list_packages()

        # Verify recursion limit was set
        assert sys.getrecursionlimit() == 10_000

        # Restore original
        sys.setrecursionlimit(original_limit)

    @patch("mngs.gen._list_packages.distributions")
    @patch("mngs.gen._list_packages.inspect_module")
    def test_hyphen_to_underscore_conversion(self, mock_inspect, mock_distributions):
        """Test that package names with hyphens are converted to underscores."""
        # Setup mocks
        mock_distributions.return_value = [MockDistribution("scikit-learn")]

        mock_inspect.return_value = pd.DataFrame({"Name": ["sklearn.test"]})

        # Call function
        list_packages()

        # Verify hyphen converted to underscore
        mock_inspect.assert_called_with(
            "scikit_learn",  # Converted from scikit-learn
            docstring=False,
            print_output=False,
            columns=["Name"],
            root_only=True,
            max_depth=1,
            skip_depwarnings=True,
        )

    def test_main_function(self):
        """Test the main function."""
        # Create DataFrame outside patch context
        test_df = pd.DataFrame({"Name": ["test.module"]})

        with patch("mngs.gen._list_packages.list_packages") as mock_list_packages:
            with patch("builtins.__import__") as mock_import:
                # Setup mocks
                mock_ipdb = MagicMock()
                mock_import.return_value = mock_ipdb
                mock_list_packages.return_value = test_df

                # Call main
                result = main()

                # Verify
                assert result == 0
                mock_list_packages.assert_called_once_with(verbose=True)
                mock_ipdb.set_trace.assert_called_once()


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
