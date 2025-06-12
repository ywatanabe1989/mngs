#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-11 03:20:00 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/_subplots/test__export_as_csv.py
=======
# Timestamp: "2025-05-18 06:06:01 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/_subplots/test__export_as_csv.py
>>>>>>> origin/main
# ----------------------------------------
"""Comprehensive tests for export_as_csv functionality."""

import os
import warnings

__FILE__ = "./tests/mngs/plt/_subplots/test__export_as_csv.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-18 06:05:04 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/_subplots/test__export_as_csv.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./tests/mngs/plt/_subplots/test__export_as_csv.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------

<<<<<<< HEAD
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from unittest.mock import patch, MagicMock

# Try direct import
try:
    from mngs.plt._subplots._export_as_csv import export_as_csv, format_record, _format_imshow2d
except ImportError:
    # Skip tests if module not properly available
    pytest.skip("Module mngs.plt._subplots._export_as_csv not available", allow_module_level=True)


class TestExportAsCSV:
    """Test suite for export_as_csv function."""
    
    def test_empty_history(self):
        """Test export with empty history."""
        history = {}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = export_as_csv(history)
            assert isinstance(result, pd.DataFrame)
            assert result.empty
            assert len(w) == 1
            assert "Plotting records not found" in str(w[0].message)

    def test_simple_plot(self):
        """Test export with simple plot."""
        history = {"plot1": ("plot1", "plot", ([1, 2, 3], [4, 5, 6]), {})}
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert "plot1_plot_x" in result.columns
        assert "plot1_plot_y" in result.columns
        assert result["plot1_plot_x"].tolist() == [1, 2, 3]
        assert result["plot1_plot_y"].tolist() == [4, 5, 6]

    def test_multiple_plots(self):
        """Test export with multiple plots."""
        history = {
            "plot1": ("plot1", "plot", ([1, 2, 3], [4, 5, 6]), {}),
            "plot2": ("plot2", "plot", ([4, 5, 6], [1, 2, 3]), {}),
        }
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [
            "plot1_plot_x",
            "plot1_plot_y",
            "plot2_plot_x",
            "plot2_plot_y",
        ]
        
    def test_export_concat_failure(self):
        """Test export when concat fails."""
        # Create a mock that raises exception
        with patch('pandas.concat', side_effect=ValueError("Test error")):
            history = {
                "plot1": ("plot1", "plot", ([1, 2], [3, 4]), {}),
                "plot2": ("plot2", "plot", ([5, 6, 7], [8, 9, 10]), {})  # Different length
            }
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = export_as_csv(history)
                assert isinstance(result, pd.DataFrame)
                assert result.empty
                assert len(w) == 1
                assert "Plotting records not combined" in str(w[0].message)

    def test_scatter_plot(self):
        """Test export with scatter plot."""
        history = {"scatter1": ("scatter1", "scatter", ([1, 2, 3], [4, 5, 6]), {})}
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert "scatter1_scatter_x" in result.columns
        assert "scatter1_scatter_y" in result.columns
        assert result["scatter1_scatter_x"].tolist() == [1, 2, 3]
        assert result["scatter1_scatter_y"].tolist() == [4, 5, 6]

    def test_bar_plot(self):
        """Test export with bar plot."""
        history = {"bar1": ("bar1", "bar", (["A", "B", "C"], [4, 5, 6]), {})}
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert "bar1_bar_x" in result.columns
        assert "bar1_bar_y" in result.columns
        assert result["bar1_bar_x"].tolist() == ["A", "B", "C"]
        assert result["bar1_bar_y"].tolist() == [4, 5, 6]

    def test_bar_plot_with_yerr(self):
        """Test export with bar plot including error bars."""
        history = {
            "bar1": (
                "bar1",
                "bar",
                (["A", "B", "C"], [4, 5, 6]),
                {"yerr": [0.1, 0.2, 0.3]},
            )
        }
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert "bar1_bar_yerr" in result.columns
        assert result["bar1_bar_yerr"].tolist() == [0.1, 0.2, 0.3]
        
    def test_histogram_plot(self):
        """Test export with histogram."""
        history = {"hist1": ("hist1", "hist", [1, 2, 2, 3, 3, 3], {})}
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert "hist1_hist_x" in result.columns
        assert result["hist1_hist_x"].tolist() == [1, 2, 2, 3, 3, 3]
        
    def test_mixed_plot_types(self):
        """Test export with mixed plot types."""
        history = {
            "plot1": ("plot1", "plot", ([1, 2], [3, 4]), {}),
            "scatter1": ("scatter1", "scatter", ([5, 6], [7, 8]), {}),
            "bar1": ("bar1", "bar", (["X", "Y"], [9, 10]), {})
        }
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 6  # 2 cols per plot type
        assert all(col in result.columns for col in [
            "plot1_plot_x", "plot1_plot_y",
            "scatter1_scatter_x", "scatter1_scatter_y", 
            "bar1_bar_x", "bar1_bar_y"
        ])


class TestFormatRecord:
    """Test suite for format_record function."""
    
    def test_imshow2d_format(self):
        """Test formatting imshow2d data."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        record = ("img1", "imshow2d", df, {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)
        
    def test_plot_with_single_array(self):
        """Test plot formatting with single 2D array."""
        record = ("plot1", "plot", [np.array([[1, 4], [2, 5], [3, 6]])], {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "plot1_plot_x" in result.columns
        assert result["plot1_plot_x"].tolist() == [1, 2, 3]

    def test_plot_with_separate_arrays(self):
        """Test plot formatting with separate x and y arrays."""
        record = (
            "plot1",
            "plot",
            [np.array([1, 2, 3]), np.array([4, 5, 6])],
            {},
        )
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "plot1_plot_x" in result.columns
        assert "plot1_plot_y" in result.columns
        assert result["plot1_plot_x"].tolist() == [1, 2, 3]
        assert result["plot1_plot_y"].tolist() == [4, 5, 6]

    def test_plot_with_2d_y_array(self):
        """Test plot formatting with 2D y array (multiple lines)."""
        record = (
            "plot1",
            "plot",
            [np.array([1, 2, 3]), np.array([[4, 7], [5, 8], [6, 9]])],
            {},
        )
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "plot1_plot_x00" in result.columns
        assert "plot1_plot_y00" in result.columns
        assert "plot1_plot_y01" in result.columns
        assert result["plot1_plot_y00"].tolist() == [4, 5, 6]
        assert result["plot1_plot_y01"].tolist() == [7, 8, 9]

    def test_plot_with_dataframe_y(self):
        """Test plot formatting with DataFrame as y values."""
        y_df = pd.DataFrame({"col1": [4, 5, 6], "col2": [7, 8, 9]})
        record = ("plot1", "plot", [np.array([1, 2, 3]), y_df], {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "plot1_plot_x" in result.columns
        assert "plot1_plot_y00" in result.columns
        assert "plot1_plot_y01" in result.columns
        
    def test_plot_with_xarray(self):
        """Test plot formatting with xarray DataArray."""
        y_xr = xr.DataArray(
            [[4, 7], [5, 8], [6, 9]], 
            dims=["x", "y"]
        )
        record = ("plot1", "plot", [np.array([1, 2, 3]), y_xr], {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "plot1_plot_x00" in result.columns
        assert "plot1_plot_y00" in result.columns
        assert "plot1_plot_y01" in result.columns
        
    def test_plot_with_list_y(self):
        """Test plot formatting with list as y values."""
        record = ("plot1", "plot", [np.array([1, 2, 3]), [4, 5, 6]], {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "plot1_plot_x" in result.columns
        assert "plot1_plot_y" in result.columns

    def test_bar_with_scalar_values(self):
        """Test bar formatting with scalar x and y."""
        record = ("bar1", "bar", (1, 5), {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "bar1_bar_x" in result.columns
        assert "bar1_bar_y" in result.columns
        assert len(result) == 1
        
    def test_bar_with_scalar_yerr(self):
        """Test bar formatting with scalar error value."""
        record = ("bar1", "bar", (["A"], [5]), {"yerr": 0.5})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "bar1_bar_yerr" in result.columns
        assert result["bar1_bar_yerr"].iloc[0] == 0.5

    def test_boxplot(self):
        """Test boxplot formatting with single box."""
        record = ("box1", "boxplot", [[1, 2, 3, 4, 5]], {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "box1_boxplot_0_x" in result.columns
        assert result["box1_boxplot_0_x"].tolist() == [1, 2, 3, 4, 5]
        
    def test_boxplot_multiple(self):
        """Test boxplot formatting with multiple boxes."""
        record = ("box1", "boxplot", [[[1, 2, 3], [4, 5, 6, 7]]], {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "box1_boxplot_0_x" in result.columns
        assert "box1_boxplot_1_x" in result.columns
        # Check dropna behavior
        assert len(result["box1_boxplot_0_x"].dropna()) == 3
        assert len(result["box1_boxplot_1_x"].dropna()) == 4
        
    def test_boxplot_with_numpy(self):
        """Test boxplot formatting with numpy array."""
        record = ("box1", "boxplot", [np.array([1.5, 2.5, 3.5])], {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
        
    def test_plot_fillv(self):
        """Test plot_fillv formatting."""
        record = ("fill1", "plot_fillv", ([1, 3, 5], [2, 4, 6]), {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "fill1_plot_fillv_starts" in result.columns
        assert "fill1_plot_fillv_ends" in result.columns
        assert result["fill1_plot_fillv_starts"].tolist() == [1, 3, 5]
        assert result["fill1_plot_fillv_ends"].tolist() == [2, 4, 6]
        
    def test_plot_raster(self):
        """Test plot_raster formatting."""
        df = pd.DataFrame({"spike_times": [0.1, 0.5, 1.2]})
        record = ("raster1", "plot_raster", df, {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)
        
    def test_plot_ecdf(self):
        """Test plot_ecdf formatting."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5]})
        record = ("ecdf1", "plot_ecdf", df, {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)
        
    def test_plot_kde(self):
        """Test plot_kde formatting."""
        df = pd.DataFrame({"density": [0.1, 0.3, 0.5, 0.3, 0.1]})
        record = ("kde1", "plot_kde", df, {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "kde1_plot_kde_density" in result.columns
        
    def test_plot_kde_no_id(self):
        """Test plot_kde formatting without ID."""
        df = pd.DataFrame({"density": [0.1, 0.3, 0.5]})
        record = (None, "plot_kde", df, {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert result.columns[0] == "density"  # Original column name preserved
        
    def test_sns_barplot(self):
        """Test seaborn barplot formatting."""
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [2, 4, 6],
            "C": [3, 6, 9]
        })
        record = ("sns_bar1", "sns_barplot", df, {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 3)  # Diagonal values
        
    def test_sns_boxplot(self):
        """Test seaborn boxplot formatting."""
        df = pd.DataFrame({
            "group1": [1, 2, 3, 4],
            "group2": [5, 6, 7, 8]
        })
        record = ("sns_box1", "sns_boxplot", df, {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "sns_box1_sns_boxplot_group1" in result.columns
        assert "sns_box1_sns_boxplot_group2" in result.columns
        
    def test_sns_boxplot_no_id(self):
        """Test seaborn boxplot formatting without ID."""
        df = pd.DataFrame({"data": [1, 2, 3]})
        record = (None, "sns_boxplot", df, {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert result.columns[0] == "data"
        
    def test_sns_heatmap(self):
        """Test seaborn heatmap formatting."""
        df = pd.DataFrame(np.random.rand(3, 3))
        record = ("heatmap1", "sns_heatmap", df, {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)
        
    def test_sns_histplot(self):
        """Test seaborn histplot formatting."""
        df = pd.DataFrame({"values": np.random.randn(100)})
        record = ("hist1", "sns_histplot", df, {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)
        
    def test_sns_violinplot(self):
        """Test seaborn violinplot formatting."""
        df = pd.DataFrame({
            "A": np.random.randn(50),
            "B": np.random.randn(50)
        })
        record = ("violin1", "sns_violinplot", df, {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)
        
    def test_unsupported_method(self):
        """Test formatting with unsupported method."""
        record = ("unknown1", "unknown_method", [1, 2, 3], {})
        result = format_record(record)
        assert result is None
        
    def test_set_method_ignored(self):
        """Test that set_ methods are ignored."""
        record = ("set1", "set_xlabel", ["X Label"], {})
        result = format_record(record)
        assert result is None


class TestFormatImshow2D:
    """Test suite for _format_imshow2d function."""
    
    def test_basic_imshow2d(self):
        """Test basic imshow2d formatting."""
        df = pd.DataFrame(np.random.rand(5, 5))
        record = ("img1", "imshow2d", df, {})
        result = _format_imshow2d(record)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)
        
    def test_imshow2d_preserves_structure(self):
        """Test that imshow2d preserves DataFrame structure."""
        df = pd.DataFrame(
            np.arange(9).reshape(3, 3),
            index=["row1", "row2", "row3"],
            columns=["col1", "col2", "col3"]
        )
        record = ("img1", "imshow2d", df, {})
        result = _format_imshow2d(record)
        pd.testing.assert_frame_equal(result, df)
        assert list(result.index) == ["row1", "row2", "row3"]
        assert list(result.columns) == ["col1", "col2", "col3"]


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_none_values_in_history(self):
        """Test handling of None values in history."""
        history = {
            "plot1": ("plot1", "plot", ([1, 2, None], [4, None, 6]), {}),
        }
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert pd.isna(result["plot1_plot_x"].iloc[2])
        assert pd.isna(result["plot1_plot_y"].iloc[1])
        
    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        history = {
            "plot1": ("plot1", "plot", ([], []), {}),
        }
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        
    def test_mismatched_array_lengths(self):
        """Test handling of mismatched array lengths in plots."""
        # This should be handled by the plotting function, but test robustness
        record = ("plot1", "plot", ([1, 2, 3], [4, 5]), {})
        # Format record should handle this gracefully
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        
    def test_unicode_in_labels(self):
        """Test handling of unicode characters in labels."""
        history = {
            "plot1": ("plot1", "bar", (["α", "β", "γ"], [1, 2, 3]), {})
        }
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert result["plot1_bar_x"].tolist() == ["α", "β", "γ"]
        
    def test_very_long_ids(self):
        """Test handling of very long plot IDs."""
        long_id = "a" * 100
        history = {
            long_id: (long_id, "plot", ([1, 2], [3, 4]), {})
        }
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert f"{long_id}_plot_x" in result.columns
        
    def test_special_characters_in_id(self):
        """Test handling of special characters in plot IDs."""
        special_id = "plot-1_test@#$"
        history = {
            special_id: (special_id, "plot", ([1, 2], [3, 4]), {})
        }
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert f"{special_id}_plot_x" in result.columns

=======
# from unittest.mock import patch

# import numpy as np
# import pandas as pd
# import pytest
# # Import functions to test from the module
# from mngs.plt._subplots._export_as_csv import export_as_csv, format_record


# class TestExportAsCSV:
#     """Test the main export_as_csv function."""

#     def test_empty_history(self):
#         """Test that empty history raises ValueError."""
#         history = {}
#         # Test with empty history_records
#         with pytest.raises(ValueError) as excinfo:
#             export_as_csv(history)
#         assert "Plotting records not found" in str(excinfo.value)

#     def test_simple_plot(self):
#         """Test export_as_csv with a single plot record."""
#         history = {"plot1": ("plot1", "plot", ([1, 2, 3], [4, 5, 6]), {})}
#         # Patch format_record to avoid any potential issues
#         with patch(
#             "mngs.plt._subplots._export_as_csv.format_record"
#         ) as mock_format:
#             mock_format.return_value = pd.DataFrame(
#                 {"plot1_plot_x": [1, 2, 3], "plot1_plot_y": [4, 5, 6]}
#             )
#             result = export_as_csv(history)
#             mock_format.assert_called_once_with(
#                 ("plot1", "plot", ([1, 2, 3], [4, 5, 6]), {})
#             )
#             assert isinstance(result, pd.DataFrame)
#             assert "plot1_plot_x" in result.columns
#             assert "plot1_plot_y" in result.columns
#             assert result["plot1_plot_x"].tolist() == [1, 2, 3]
#             assert result["plot1_plot_y"].tolist() == [4, 5, 6]

#     def test_multiple_plots(self):
#         """Test export_as_csv with multiple plot records."""
#         history = {
#             "plot1": ("plot1", "plot", ([1, 2, 3], [4, 5, 6]), {}),
#             "plot2": ("plot2", "plot", ([4, 5, 6], [1, 2, 3]), {}),
#         }
#         # Patch format_record
#         with patch(
#             "mngs.plt._subplots._export_as_csv.format_record"
#         ) as mock_format:
#             mock_format.side_effect = [
#                 pd.DataFrame(
#                     {"plot1_plot_x": [1, 2, 3], "plot1_plot_y": [4, 5, 6]}
#                 ),
#                 pd.DataFrame(
#                     {"plot2_plot_x": [4, 5, 6], "plot2_plot_y": [1, 2, 3]}
#                 ),
#             ]
#             result = export_as_csv(history)
#             assert mock_format.call_count == 2
#             assert isinstance(result, pd.DataFrame)
#             assert list(result.columns) == [
#                 "plot1_plot_x",
#                 "plot1_plot_y",
#                 "plot2_plot_x",
#                 "plot2_plot_y",
#             ]

#     # def test_concat_failure(self):
#     #     """Test that export_as_csv raises ValueError on DataFrame concatenation failures."""
#     #     history = {
#     #         "plot1": ("plot1", "plot", ([1, 2, 3], [4, 5, 6]), {}),
#     #         "bar1": ("bar1", "bar", (["A", "B", "C"], [4, 5, 6]), {}),
#     #     }
#     #     # Mock format_record to simulate a situation where concat fails
#     #     with patch(
#     #         "mngs.plt._subplots._export_as_csv.format_record"
#     #     ) as mock_format:
#     #         # Return a DataFrame and then something that can't be concatenated with it
#     #         mock_format.side_effect = [
#     #             pd.DataFrame(
#     #                 {"plot1_plot_x": [1, 2, 3], "plot1_plot_y": [4, 5, 6]}
#     #             ),
#     #             pd.DataFrame(
#     #                 {"incompatible": ["A", "B"]}
#     #             ),  # Dataframe with different length
#     #         ]
#     #         with pytest.raises(ValueError) as excinfo:
#     #             export_as_csv(history)
#     #         assert "Failed to combine plotting records" in str(excinfo.value)


# class TestFormatRecord:
#     """Test the format_record function that routes to specific formatting functions."""

#     def test_dispatch_to_formatters(self):
#         """Test that format_record correctly dispatches to the appropriate formatters."""
#         # Test a variety of record types to ensure proper routing
#         with patch(
#             "mngs.plt._subplots._export_as_csv.format_plot"
#         ) as mock_plot:
#             mock_plot.return_value = pd.DataFrame({"test": [1]})
#             result = format_record(("id1", "plot", ([1, 2], [3, 4]), {}))
#             mock_plot.assert_called_once_with("id1", ([1, 2], [3, 4]), {})
#             assert isinstance(result, pd.DataFrame)

#         with patch("mngs.plt._subplots._export_as_csv.format_bar") as mock_bar:
#             mock_bar.return_value = pd.DataFrame({"test": [1]})
#             result = format_record(("id2", "bar", ([1, 2], [3, 4]), {}))
#             mock_bar.assert_called_once_with("id2", ([1, 2], [3, 4]), {})
#             assert isinstance(result, pd.DataFrame)

#         with patch(
#             "mngs.plt._subplots._export_as_csv.format_sns_barplot"
#         ) as mock_sns:
#             mock_sns.return_value = pd.DataFrame({"test": [1]})
#             result = format_record(("id3", "sns_barplot", pd.DataFrame(), {}))
#             mock_sns.assert_called_once_with("id3", pd.DataFrame(), {})
#             assert isinstance(result, pd.DataFrame)

#     def test_unknown_method(self):
#         """Test that format_record raises NotImplementedError for unknown methods."""
#         with pytest.raises(NotImplementedError) as excinfo:
#             format_record(("id", "unknown_method", [], {}))
#         assert "is not yet implemented" in str(excinfo.value)
#         assert "unknown_method" in str(excinfo.value)


# class TestFormatRecordAsFacade:
#     """Test the format_record function acting as a facade for the various formatters."""

#     def test_format_record_with_plot(self):
#         """Test format_record with plot data."""
#         record = ("plot1", "plot", ([1, 2, 3], [4, 5, 6]), {})
#         result = format_record(record)
#         assert isinstance(result, pd.DataFrame)
#         assert "plot1_plot_x" in result.columns
#         assert "plot1_plot_y" in result.columns
#         assert result["plot1_plot_x"].tolist() == [1, 2, 3]
#         assert result["plot1_plot_y"].tolist() == [4, 5, 6]

#     def test_format_record_with_scatter(self):
#         """Test format_record with scatter data."""
#         record = ("scatter1", "scatter", ([1, 2, 3], [4, 5, 6]), {})
#         result = format_record(record)
#         assert isinstance(result, pd.DataFrame)
#         assert "scatter1_scatter_x" in result.columns
#         assert "scatter1_scatter_y" in result.columns
#         assert result["scatter1_scatter_x"].tolist() == [1, 2, 3]
#         assert result["scatter1_scatter_y"].tolist() == [4, 5, 6]

#     def test_format_record_with_bar(self):
#         """Test format_record with bar data."""
#         record = ("bar1", "bar", (["A", "B", "C"], [4, 5, 6]), {})
#         result = format_record(record)
#         assert isinstance(result, pd.DataFrame)
#         assert "bar1_bar_x" in result.columns
#         assert "bar1_bar_y" in result.columns
#         assert result["bar1_bar_x"].tolist() == ["A", "B", "C"]
#         assert result["bar1_bar_y"].tolist() == [4, 5, 6]

#     def test_format_record_with_hist(self):
#         """Test format_record with hist data."""
#         data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
#         record = ("hist1", "hist", data, {})
#         result = format_record(record)
#         assert isinstance(result, pd.DataFrame)
#         assert "hist1_hist_x" in result.columns
#         assert result["hist1_hist_x"].tolist() == data

#     def test_format_record_with_boxplot(self):
#         """Test format_record with boxplot data."""
#         data = [1, 2, 3, 4, 5]
#         record = ("box1", "boxplot", [data], {})

#         # Since we need to patch mngs.types.is_listed_X
#         with patch("mngs.types.is_listed_X") as mock_is_listed_X:
#             mock_is_listed_X.return_value = True
#             result = format_record(record)
#             assert isinstance(result, pd.DataFrame)
#             assert "box1_boxplot_0_x" in result.columns

#     def test_format_record_with_barh(self):
#         """Test format_record with barh data."""
#         record = ("barh1", "barh", (["A", "B", "C"], [4, 5, 6]), {})
#         result = format_record(record)
#         assert isinstance(result, pd.DataFrame)
#         assert "barh1_barh_x" in result.columns
#         assert "barh1_barh_y" in result.columns
#         assert result["barh1_barh_y"].tolist() == ["A", "B", "C"]
#         assert result["barh1_barh_x"].tolist() == [4, 5, 6]

#     def test_format_record_with_fill_between(self):
#         """Test format_record with fill_between data."""
#         x = [1, 2, 3, 4, 5]
#         y1 = [1, 3, 2, 4, 2]
#         y2 = [3, 4, 3, 5, 3]
#         record = ("fillbw1", "fill_between", [x, y1, y2], {})
#         result = format_record(record)
#         assert isinstance(result, pd.DataFrame)
#         assert "fillbw1_fill_between_x" in result.columns
#         assert "fillbw1_fill_between_y1" in result.columns
#         assert "fillbw1_fill_between_y2" in result.columns

#     def test_format_record_with_contour(self):
#         """Test format_record with contour data."""
#         # Create sample mesh grid data
#         x = np.linspace(-3, 3, 5)
#         y = np.linspace(-3, 3, 5)
#         X, Y = np.meshgrid(x, y)
#         Z = np.sin(X) + np.cos(Y)
#         record = ("contour1", "contour", [X, Y, Z], {})

#         result = format_record(record)
#         assert isinstance(result, pd.DataFrame)
#         assert "contour1_contour_x" in result.columns
#         assert "contour1_contour_y" in result.columns
#         assert "contour1_contour_z" in result.columns

#     def test_format_record_with_plot_line(self):
#         """Test format_record with plot_line data."""
#         x = np.array([1, 2, 3, 4, 5])
#         y = np.array([5, 4, 3, 2, 1])
#         record = ("line1", "plot_line", [x, y], {})

#         result = format_record(record)
#         assert isinstance(result, pd.DataFrame)
#         assert "line1_plot_line_x" in result.columns
#         assert "line1_plot_line_y" in result.columns
#         assert result["line1_plot_line_x"].tolist() == [1, 2, 3, 4, 5]
#         assert result["line1_plot_line_y"].tolist() == [5, 4, 3, 2, 1]

#     def test_format_record_with_plot_line_2d(self):
#         """Test format_record with plot_line data with 2D y values."""
#         x = np.array([1, 2, 3, 4, 5])
#         y = np.array([[5, 10], [4, 8], [3, 6], [2, 4], [1, 2]])
#         record = ("line2", "plot_line", [x, y], {})

#         result = format_record(record)
#         assert isinstance(result, pd.DataFrame)
#         assert "line2_plot_line_x" in result.columns
#         assert "line2_plot_line_y00" in result.columns
#         assert "line2_plot_line_y01" in result.columns
#         assert result["line2_plot_line_x"].tolist() == [1, 2, 3, 4, 5]
#         assert result["line2_plot_line_y00"].tolist() == [5, 4, 3, 2, 1]
#         assert result["line2_plot_line_y01"].tolist() == [10, 8, 6, 4, 2]

#     def test_format_record_with_sns_scatterplot(self):
#         """Test format_record with sns_scatterplot data."""
#         # Test with DataFrame input
#         df = pd.DataFrame(
#             {
#                 "x": [1, 2, 3, 4, 5],
#                 "y": [5, 4, 3, 2, 1],
#                 "group": ["A", "A", "B", "B", "C"],
#             }
#         )
#         record = (
#             "sns_scatter1",
#             "sns_scatterplot",
#             [df],
#             {"x": "x", "y": "y", "hue": "group"},
#         )

#         result = format_record(record)
#         assert isinstance(result, pd.DataFrame)
#         assert "sns_scatter1_scatter_x" in result.columns
#         assert "sns_scatter1_scatter_y" in result.columns
#         assert "sns_scatter1_scatter_group" in result.columns
#         assert result["sns_scatter1_scatter_x"].tolist() == [1, 2, 3, 4, 5]
#         assert result["sns_scatter1_scatter_y"].tolist() == [5, 4, 3, 2, 1]
#         assert result["sns_scatter1_scatter_group"].tolist() == [
#             "A",
#             "A",
#             "B",
#             "B",
#             "C",
#         ]

#     def test_format_record_with_sns_lineplot(self):
#         """Test format_record with sns_lineplot data."""
#         # Test with direct x, y array input
#         x = np.array([1, 2, 3, 4, 5])
#         y = np.array([5, 4, 3, 2, 1])
#         record = ("sns_line1", "sns_lineplot", [x, y], {})

#         result = format_record(record)
#         assert isinstance(result, pd.DataFrame)
#         assert "sns_line1_line_x" in result.columns
#         assert "sns_line1_line_y" in result.columns
#         assert result["sns_line1_line_x"].tolist() == [1, 2, 3, 4, 5]
#         assert result["sns_line1_line_y"].tolist() == [5, 4, 3, 2, 1]
>>>>>>> origin/main

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/_subplots/_export_as_csv.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-18 00:23:26 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_export_as_csv.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_subplots/_export_as_csv.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# import sys
# import warnings
<<<<<<< HEAD
#
=======
# from collections import OrderedDict
# 
>>>>>>> origin/main
# import matplotlib.pyplot as plt
# import mngs
# import numpy as np
# import pandas as pd
# import xarray as xr
<<<<<<< HEAD
#
#
=======
# 
# # ----------------------------------------
# # Main function to export records as CSV
# # ----------------------------------------
# 
>>>>>>> origin/main
# def export_as_csv(history_records):
#     """Convert plotting history records to a combined DataFrame suitable for CSV export.
# 
#     Args:
#         history_records (dict): Dictionary of plotting records.
# 
#     Returns:
#         pd.DataFrame: Combined DataFrame containing all plotting data.
#         
#     Raises:
#         ValueError: If no plotting records are found or they cannot be combined.
#     """
#     if len(history_records) <= 0:
<<<<<<< HEAD
#         warnings.warn("Plotting records not found. Empty dataframe returned.")
#         return pd.DataFrame()
#     else:
#         dfs = [
#             format_record(record) for record in list(history_records.values())
#         ]
#         try:
#             df = pd.concat(dfs, axis=1)
#             return df
#         except Exception as e:
#             warnings.warn(
#                 f"Plotting records not combined. Empty dataframe returned {e}"
#             )
#             return pd.DataFrame()
#
#
# def _format_imshow2d(record):
#     id, method, args, kwargs = record
#     df = args
#     # df.columns = [f"{id}_{method}_{col}" for col in df.columns]
#     # df.index = [f"{id}_{method}_{idx}" for idx in df.index]
#     return df
#
#
=======
#         raise ValueError("Plotting records not found. Cannot export empty data.")
#     
#     dfs = [format_record(record) for record in list(history_records.values())]
#     
#     if all(df.empty for df in dfs):
#         raise ValueError("All records resulted in empty dataframes. Cannot export data.")
#         
#     try:
#         df = pd.concat(dfs, axis=1)
#         return df
#     except Exception as e:
#         raise ValueError(f"Failed to combine plotting records into a dataframe: {e}")
# 
# 
>>>>>>> origin/main
# def format_record(record):
#     """Route record to the appropriate formatting function based on plot method.
# 
#     Args:
#         record (tuple): Plotting record tuple (id, method, args, kwargs).
# 
#     Returns:
#         pd.DataFrame: Formatted data for the plot record.
#     """
#     id, method, args, kwargs = record
<<<<<<< HEAD
#
#     if method == "imshow2d":
#         return _format_imshow2d(record)
#
#     elif method in ["plot"]:
#         if len(args) == 1:
#             args = args[0]
#             if args.ndim == 2:
#                 x, y = args[:, 0], args[:, 1]
#                 df = pd.DataFrame({f"{id}_{method}_x": x})
#                 return df
#
#         elif len(args) == 2:
#             x, y = args
#
#             if isinstance(y, (np.ndarray, xr.DataArray)):
#                 if y.ndim == 2:
#                     from collections import OrderedDict
#
#                     out = OrderedDict()
#
#                     for ii in range(y.shape[1]):
#                         out[f"{id}_{method}_x{ii:02d}"] = x
#                         out[f"{id}_{method}_y{ii:02d}"] = y[:, ii]
#                     df = pd.DataFrame(out)
#
#                     return df
#
#             if isinstance(y, pd.DataFrame):
#                 df = pd.DataFrame(
#                     {
#                         f"{id}_{method}_x": x,
#                         **{
#                             f"{id}_{method}_y{ii:02d}": np.array(y[col])
#                             for ii, col in enumerate(y.columns)
#                         },
#                     }
#                 )
#                 return df
#
#             else:
#                 if isinstance(y, (np.ndarray, xr.DataArray, list)):
#                     df = pd.DataFrame(
#                         {f"{id}_{method}_x": x, f"{id}_{method}_y": y}
#                     )
#                     return df
#
#     elif method == "scatter":
#         x, y = args
#         df = pd.DataFrame({f"{id}_{method}_x": x, f"{id}_{method}_y": y})
#         return df
#
#     elif method == "bar":
#         x, y = args
#         yerr = kwargs.get("yerr")
#
#         if isinstance(x, (int, float)):
#             x = pd.Series(x, name="x")
#         if isinstance(y, (int, float)):
#             y = pd.Series(y, name="y")
#
#         df = pd.DataFrame({f"{id}_{method}_x": x, f"{id}_{method}_y": y})
#
#         if yerr is not None:
#             if isinstance(yerr, (int, float)):
#                 yerr = pd.Series(yerr, name="yerr")
#             df[f"{id}_{method}_yerr"] = yerr
#         return df
#
#     elif method == "hist":
#         x = args
#         df = pd.DataFrame({f"{id}_{method}_x": x})
#         return df
#
#     elif method == "boxplot":
#         x = args[0]
#
#         # One box plot
#         from mngs.types import is_listed_X as mngs_types_is_listed_X
#
#         if isinstance(x, np.ndarray) or mngs_types_is_listed_X(
#             x, [float, int]
#         ):
#             df = pd.DataFrame(x)
#
#         else:
#             # Multiple boxes
#             import mngs.pd.force_df as mngs_pd_force_df
#
#             df = mngs.pd.force_df({i_x: _x for i_x, _x in enumerate(x)})
#         df.columns = [f"{id}_{method}_{col}_x" for col in df.columns]
#         df = df.apply(lambda col: col.dropna().reset_index(drop=True))
#         return df
#
#     # elif method == "boxplot_":
#     #     __import__("ipdb").set_trace()
#     #     x = args[0]
#     #     df =
#     #     df.columns = [f"{id}_{method}_{col}" for col in df.columns]
#
#     #     return df
#
#     # elif method == "plot_":
#     #     df = args
#     #     df.columns = [f"{id}_{method}_{col}" for col in df.columns]
#     #     return df
#
#     elif method == "plot_fillv":
#         starts, ends = args
#         df = pd.DataFrame(
#             {
#                 f"{id}_{method}_starts": starts,
#                 f"{id}_{method}_ends": ends,
#             }
#         )
#         return df
#
#     elif method == "plot_raster":
#         df = args
#         return df
#
#     elif method == "plot_ecdf":
#         df = args
#         return df
#
#     elif method == "plot_kde":
#         df = args
#         if id is not None:
#             df.columns = [f"{id}_{method}_{col}" for col in df.columns]
#         return df
#
#     elif method == "sns_barplot":
#         df = args
#
#         # When xyhue, without errorbar
#         df = pd.DataFrame(
#             pd.Series(np.array(df).diagonal(), index=df.columns)
#         ).T
#         return df
#
#     elif method == "sns_boxplot":
#         df = args
#         if id is not None:
#             df.columns = [f"{id}_{method}_{col}" for col in df.columns]
#         return df
#
#     elif method == "sns_heatmap":
#         df = args
#         return df
#
#     elif method == "sns_histplot":
#         df = args
#         return df
#
#     elif method == "sns_kdeplot":
#         pass
#         # df = args
#         # __import__("ipdb").set_trace()
#         # return df
#
#     elif method == "sns_lineplot":
#         __import__("ipdb").set_trace()
#         return df
#
#     elif method == "sns_pairplot":
#         __import__("ipdb").set_trace()
#         return df
#
#     elif method == "sns_scatterplot":
#         return df
#
#     elif method == "sns_violinplot":
#         df = args
#         return df
#
#     elif method == "sns_jointplot":
#         __import__("ipdb").set_trace()
#         return df
#
#     else:
#         pass
#         # if not method.startswith("set_"):
#         #     logging.warn(
#         #         f"{method} is not implemented in _export_as_csv method of the mngs.plt module."
#         #     )
#
#
=======
# 
#     # Basic Matplotlib functions
#     if method == "plot":
#         return format_plot(id, args, kwargs)
#     elif method == "scatter":
#         return format_scatter(id, args, kwargs)
#     elif method == "bar":
#         return format_bar(id, args, kwargs)
#     elif method == "barh":
#         return format_barh(id, args, kwargs)
#     elif method == "hist":
#         return format_hist(id, args, kwargs)
#     elif method == "boxplot":
#         return format_boxplot(id, args, kwargs)
#     elif method == "contour":
#         return format_contour(id, args, kwargs)
#     elif method == "errorbar":
#         return format_errorbar(id, args, kwargs)
#     elif method == "eventplot":
#         return format_eventplot(id, args, kwargs)
#     elif method == "fill":
#         return format_fill(id, args, kwargs)
#     elif method == "fill_between":
#         return format_fill_between(id, args, kwargs)
#     elif method == "imshow":
#         return format_imshow(id, args, kwargs)
#     elif method == "imshow2d":
#         return format_imshow2d(id, args, kwargs)
#     elif method == "violin":
#         return format_violin(id, args, kwargs)
#     elif method == "violinplot":
#         return format_violinplot(id, args, kwargs)
# 
#     # Custom plotting functions
#     elif method == "plot_box":
#         return format_plot_box(id, args, kwargs)
#     elif method == "plot_conf_mat":
#         return format_plot_conf_mat(id, args, kwargs)
#     elif method == "plot_ecdf":
#         return format_plot_ecdf(id, args, kwargs)
#     elif method == "plot_fillv":
#         return format_plot_fillv(id, args, kwargs)
#     elif method == "plot_heatmap":
#         return format_plot_heatmap(id, args, kwargs)
#     elif method == "plot_image":
#         return format_plot_image(id, args, kwargs)
#     elif method == "plot_joyplot":
#         return format_plot_joyplot(id, args, kwargs)
#     elif method == "plot_kde":
#         return format_plot_kde(id, args, kwargs)
#     elif method == "plot_line":
#         return format_plot_line(id, args, kwargs)
#     elif method == "plot_mean_ci":
#         return format_plot_mean_ci(id, args, kwargs)
#     elif method == "plot_mean_std":
#         return format_plot_mean_std(id, args, kwargs)
#     elif method == "plot_median_iqr":
#         return format_plot_median_iqr(id, args, kwargs)
#     elif method == "plot_raster":
#         return format_plot_raster(id, args, kwargs)
#     elif method == "plot_rectangle":
#         return format_plot_rectangle(id, args, kwargs)
#     elif method == "plot_scatter_hist":
#         return format_plot_scatter_hist(id, args, kwargs)
#     elif method == "plot_shaded_line":
#         return format_plot_shaded_line(id, args, kwargs)
#     elif method == "plot_violin":
#         return format_plot_violin(id, args, kwargs)
# 
#     # Seaborn functions
#     elif method == "sns_barplot":
#         return format_sns_barplot(id, args, kwargs)
#     elif method == "sns_boxplot":
#         return format_sns_boxplot(id, args, kwargs)
#     elif method == "sns_heatmap":
#         return format_sns_heatmap(id, args, kwargs)
#     elif method == "sns_histplot":
#         return format_sns_histplot(id, args, kwargs)
#     elif method == "sns_jointplot":
#         return format_sns_jointplot(id, args, kwargs)
#     elif method == "sns_kdeplot":
#         return format_sns_kdeplot(id, args, kwargs)
#     elif method == "sns_lineplot":
#         return format_sns_lineplot(id, args, kwargs)
#     elif method == "sns_pairplot":
#         return format_sns_pairplot(id, args, kwargs)
#     elif method == "sns_scatterplot":
#         return format_sns_scatterplot(id, args, kwargs)
#     elif method == "sns_stripplot":
#         return format_sns_stripplot(id, args, kwargs)
#     elif method == "sns_swarmplot":
#         return format_sns_swarmplot(id, args, kwargs)
#     elif method == "sns_violinplot":
#         return format_sns_violinplot(id, args, kwargs)
#     else:
#         # Unknown or unimplemented method
#         raise NotImplementedError(
#             f"CSV export for plot method '{method}' is not yet implemented in the mngs.plt module. "
#             f"Check the feature-request-export-as-csv-functions.md for implementation status."
#         )
# 
# # ----------------------------------------
# # Matplotlib plotting function formatters
# # ----------------------------------------
# 
# def format_plot(id, args, kwargs):
#     """Format data from a plot call.
#     
#     Args:
#         id (str): Identifier for the plot
#         args (tuple): Arguments passed to plot
#         kwargs (dict): Keyword arguments passed to plot
#         
#     Returns:
#         pd.DataFrame: Formatted data from plot
#     """
#     if len(args) == 1:
#         args = args[0]
#         if args.ndim == 2:
#             x, y = args[:, 0], args[:, 1]
#             df = pd.DataFrame({f"{id}_plot_x": x, f"{id}_plot_y": y})
#             return df
# 
#     elif len(args) == 2:
#         x, y = args
# 
#         if isinstance(y, (np.ndarray, xr.DataArray)):
#             if y.ndim == 2:
#                 out = OrderedDict()
# 
#                 for ii in range(y.shape[1]):
#                     out[f"{id}_plot_x{ii:02d}"] = x
#                     out[f"{id}_plot_y{ii:02d}"] = y[:, ii]
#                 df = pd.DataFrame(out)
# 
#                 return df
# 
#         if isinstance(y, pd.DataFrame):
#             df = pd.DataFrame(
#                 {
#                     f"{id}_plot_x": x,
#                     **{
#                         f"{id}_plot_y{ii:02d}": np.array(y[col])
#                         for ii, col in enumerate(y.columns)
#                     },
#                 }
#             )
#             return df
# 
#         else:
#             if isinstance(y, (np.ndarray, xr.DataArray, list)):
#                 df = pd.DataFrame(
#                     {f"{id}_plot_x": x, f"{id}_plot_y": y}
#                 )
#                 return df
#                 
#     # Default empty DataFrame if we can't process the input
#     return pd.DataFrame()
# 
# def format_scatter(id, args, kwargs):
#     """Format data from a scatter call."""
#     x, y = args
#     df = pd.DataFrame({f"{id}_scatter_x": x, f"{id}_scatter_y": y})
#     return df
# 
# def format_bar(id, args, kwargs):
#     """Format data from a bar call."""
#     x, y = args
#     yerr = kwargs.get("yerr")
# 
#     if isinstance(x, (int, float)):
#         x = pd.Series(x, name="x")
#     if isinstance(y, (int, float)):
#         y = pd.Series(y, name="y")
# 
#     df = pd.DataFrame({f"{id}_bar_x": x, f"{id}_bar_y": y})
# 
#     if yerr is not None:
#         if isinstance(yerr, (int, float)):
#             yerr = pd.Series(yerr, name="yerr")
#         df[f"{id}_bar_yerr"] = yerr
#     return df
# 
# def format_barh(id, args, kwargs):
#     """Format data from a barh call."""
#     x, y = args  # Note: in barh, x is height, y is width (visually transposed from bar)
#     xerr = kwargs.get("xerr")
# 
#     if isinstance(x, (int, float)):
#         x = pd.Series(x, name="x")
#     if isinstance(y, (int, float)):
#         y = pd.Series(y, name="y")
# 
#     df = pd.DataFrame({f"{id}_barh_y": x, f"{id}_barh_x": y})  # Swap x/y for barh
# 
#     if xerr is not None:
#         if isinstance(xerr, (int, float)):
#             xerr = pd.Series(xerr, name="xerr")
#         df[f"{id}_barh_xerr"] = xerr
#     return df
# 
# def format_hist(id, args, kwargs):
#     """Format data from a hist call."""
#     x = args
#     df = pd.DataFrame({f"{id}_hist_x": x})
#     return df
# 
# def format_boxplot(id, args, kwargs):
#     """Format data from a boxplot call."""
#     x = args[0]
# 
#     # One box plot
#     from mngs.types import is_listed_X as mngs_types_is_listed_X
# 
#     if isinstance(x, np.ndarray) or mngs_types_is_listed_X(
#         x, [float, int]
#     ):
#         df = pd.DataFrame(x)
#     else:
#         # Multiple boxes
#         import mngs.pd.force_df as mngs_pd_force_df
# 
#         df = mngs.pd.force_df({i_x: _x for i_x, _x in enumerate(x)})
#     df.columns = [f"{id}_boxplot_{col}_x" for col in df.columns]
#     df = df.apply(lambda col: col.dropna().reset_index(drop=True))
#     return df
# 
# def format_contour(id, args, kwargs):
#     """Format data from a contour call."""
#     # Placeholder implementation
#     # Typical args: X, Y, Z where X and Y are 2D coordinate arrays and Z is the height array
#     if len(args) >= 3:
#         X, Y, Z = args[:3]
#         # Convert mesh grids to column vectors for export
#         X_flat = X.flatten()
#         Y_flat = Y.flatten()
#         Z_flat = Z.flatten()
#         df = pd.DataFrame({
#             f"{id}_contour_x": X_flat,
#             f"{id}_contour_y": Y_flat,
#             f"{id}_contour_z": Z_flat
#         })
#         return df
#     return pd.DataFrame()
# 
# def format_errorbar(id, args, kwargs):
#     """Format data from an errorbar call."""
#     # Typical args: x, y
#     # Typical kwargs: xerr, yerr
#     if len(args) >= 2:
#         x, y = args[:2]
#         xerr = kwargs.get('xerr')
#         yerr = kwargs.get('yerr')
#         
#         try:
#             # Try using mngs.pd.force_df if available
#             try:
#                 import mngs.pd
#                 data = {f"{id}_errorbar_x": x, f"{id}_errorbar_y": y}
#                 
#                 if xerr is not None:
#                     if isinstance(xerr, (list, tuple)) and len(xerr) == 2:
#                         # Asymmetric error
#                         data[f"{id}_errorbar_xerr_neg"] = xerr[0]
#                         data[f"{id}_errorbar_xerr_pos"] = xerr[1]
#                     else:
#                         # Symmetric error
#                         data[f"{id}_errorbar_xerr"] = xerr
#                         
#                 if yerr is not None:
#                     if isinstance(yerr, (list, tuple)) and len(yerr) == 2:
#                         # Asymmetric error
#                         data[f"{id}_errorbar_yerr_neg"] = yerr[0]
#                         data[f"{id}_errorbar_yerr_pos"] = yerr[1]
#                     else:
#                         # Symmetric error
#                         data[f"{id}_errorbar_yerr"] = yerr
#                         
#                 # Use mngs.pd.force_df to handle different length arrays
#                 df = mngs.pd.force_df(data)
#                 return df
#             except (ImportError, AttributeError):
#                 # Fall back to pandas with manual padding
#                 max_len = max([len(arr) if hasattr(arr, '__len__') else 1 
#                               for arr in [x, y, xerr, yerr] if arr is not None])
#                 
#                 # Function to pad arrays to the same length
#                 def pad_to_length(arr, length):
#                     if arr is None:
#                         return None
#                     if not hasattr(arr, '__len__'):
#                         # Handle scalar values
#                         return [arr] * length
#                     if len(arr) >= length:
#                         return arr
#                     # Pad with NaN
#                     return np.pad(arr, (0, length - len(arr)), 
#                                   'constant', constant_values=np.nan)
#                 
#                 # Pad all arrays
#                 x_padded = pad_to_length(x, max_len)
#                 y_padded = pad_to_length(y, max_len)
#                 
#                 data = {f"{id}_errorbar_x": x_padded, f"{id}_errorbar_y": y_padded}
#                 
#                 if xerr is not None:
#                     if isinstance(xerr, (list, tuple)) and len(xerr) == 2:
#                         xerr_neg_padded = pad_to_length(xerr[0], max_len)
#                         xerr_pos_padded = pad_to_length(xerr[1], max_len)
#                         data[f"{id}_errorbar_xerr_neg"] = xerr_neg_padded
#                         data[f"{id}_errorbar_xerr_pos"] = xerr_pos_padded
#                     else:
#                         xerr_padded = pad_to_length(xerr, max_len)
#                         data[f"{id}_errorbar_xerr"] = xerr_padded
#                         
#                 if yerr is not None:
#                     if isinstance(yerr, (list, tuple)) and len(yerr) == 2:
#                         yerr_neg_padded = pad_to_length(yerr[0], max_len)
#                         yerr_pos_padded = pad_to_length(yerr[1], max_len)
#                         data[f"{id}_errorbar_yerr_neg"] = yerr_neg_padded
#                         data[f"{id}_errorbar_yerr_pos"] = yerr_pos_padded
#                     else:
#                         yerr_padded = pad_to_length(yerr, max_len)
#                         data[f"{id}_errorbar_yerr"] = yerr_padded
#                 
#                 return pd.DataFrame(data)
#         except Exception as e:
#             # If all else fails, return an empty DataFrame
#             import warnings
#             warnings.warn(f"Error formatting errorbar data: {str(e)}")
#             return pd.DataFrame()
#             
#     return pd.DataFrame()
# 
# def format_eventplot(id, args, kwargs):
#     """Format data from an eventplot call."""
#     # Eventplot displays multiple sets of events as parallel lines
#     if len(args) >= 1:
#         positions = args[0]
#         
#         try:
#             # Try using mngs.pd.force_df if available
#             try:
#                 import mngs.pd
#                 
#                 # If positions is a single array
#                 if isinstance(positions, (list, np.ndarray)) and not isinstance(positions[0], (list, np.ndarray)):
#                     return pd.DataFrame({f"{id}_eventplot_events": positions})
#                     
#                 # If positions is a list of arrays (multiple event sets)
#                 elif isinstance(positions, (list, np.ndarray)):
#                     data = {}
#                     for i, events in enumerate(positions):
#                         data[f"{id}_eventplot_events{i:02d}"] = events
#                     
#                     # Use force_df to handle different length arrays
#                     return mngs.pd.force_df(data)
#                     
#             except (ImportError, AttributeError):
#                 # Fall back to pandas with manual Series creation
#                 # If positions is a single array
#                 if isinstance(positions, (list, np.ndarray)) and not isinstance(positions[0], (list, np.ndarray)):
#                     return pd.DataFrame({f"{id}_eventplot_events": positions})
#                     
#                 # If positions is a list of arrays (multiple event sets)
#                 elif isinstance(positions, (list, np.ndarray)):
#                     # Create a DataFrame where each column is a Series that can handle varying lengths
#                     df = pd.DataFrame()
#                     for i, events in enumerate(positions):
#                         df[f"{id}_eventplot_events{i:02d}"] = pd.Series(events)
#                     return df
#         except Exception as e:
#             # If all else fails, return an empty DataFrame
#             import warnings
#             warnings.warn(f"Error formatting eventplot data: {str(e)}")
#             return pd.DataFrame()
#     
#     return pd.DataFrame()
# 
# def format_fill(id, args, kwargs):
#     """Format data from a fill call."""
#     # Placeholder implementation
#     # Fill creates a polygon based on points
#     if len(args) >= 2:
#         # First arg is x, remaining args are y values
#         x = args[0]
#         data = {f"{id}_fill_x": x}
#         
#         for i, y in enumerate(args[1:]):
#             data[f"{id}_fill_y{i:02d}"] = y
#             
#         return pd.DataFrame(data)
#     return pd.DataFrame()
# 
# def format_fill_between(id, args, kwargs):
#     """Format data from a fill_between call."""
#     # Placeholder implementation
#     # Typical args: x, y1, y2
#     if len(args) >= 3:
#         x, y1, y2 = args[:3]
#         
#         df = pd.DataFrame({
#             f"{id}_fill_between_x": x,
#             f"{id}_fill_between_y1": y1,
#             f"{id}_fill_between_y2": y2
#         })
#         return df
#     return pd.DataFrame()
# 
# def format_imshow(id, args, kwargs):
#     """Format data from an imshow call."""
#     # Placeholder implementation
#     # Imshow displays an image (2D array)
#     if len(args) >= 1:
#         img = args[0]
#         
#         # Convert 2D image to long format
#         if isinstance(img, np.ndarray) and img.ndim == 2:
#             rows, cols = img.shape
#             row_indices, col_indices = np.meshgrid(range(rows), range(cols), indexing='ij')
#             
#             df = pd.DataFrame({
#                 f"{id}_imshow_row": row_indices.flatten(),
#                 f"{id}_imshow_col": col_indices.flatten(),
#                 f"{id}_imshow_value": img.flatten()
#             })
#             return df
#             
#         # Handle RGB/RGBA images
#         elif isinstance(img, np.ndarray) and img.ndim == 3:
#             rows, cols, channels = img.shape
#             row_indices, col_indices = np.meshgrid(range(rows), range(cols), indexing='ij')
#             
#             data = {
#                 f"{id}_imshow_row": row_indices.flatten(),
#                 f"{id}_imshow_col": col_indices.flatten(),
#             }
#             
#             # Add channel data
#             for c in range(channels):
#                 data[f"{id}_imshow_channel{c}"] = img[:,:,c].flatten()
#                 
#             return pd.DataFrame(data)
#     return pd.DataFrame()
# 
# def format_imshow2d(id, args, kwargs):
#     """Format data from an imshow2d call."""
#     df = args
#     # df.columns = [f"{id}_imshow2d_{col}" for col in df.columns]
#     # df.index = [f"{id}_imshow2d_{idx}" for idx in df.index]
#     return df
# 
# def format_violin(id, args, kwargs):
#     """Format data from a violin call."""
#     # Placeholder implementation
#     # Similar to boxplot but shows probability density
#     if len(args) >= 1:
#         data = args[0]
#         
#         if isinstance(data, (list, np.ndarray)):
#             return pd.DataFrame({f"{id}_violin_values": data})
#         elif isinstance(data, dict):
#             result = pd.DataFrame()
#             for label, values in data.items():
#                 result[f"{id}_violin_{label}"] = pd.Series(values)
#             return result
#     return pd.DataFrame()
# 
# def format_violinplot(id, args, kwargs):
#     """Format data from a violinplot call."""
#     # Placeholder implementation - similar to violin
#     if len(args) >= 1:
#         data = args[0]
#         
#         if isinstance(data, (list, np.ndarray)):
#             return pd.DataFrame({f"{id}_violinplot_values": data})
#         elif isinstance(data, dict):
#             result = pd.DataFrame()
#             for label, values in data.items():
#                 result[f"{id}_violinplot_{label}"] = pd.Series(values)
#             return result
#     return pd.DataFrame()
# 
# # ----------------------------------------
# # Custom plotting function formatters
# # ----------------------------------------
# 
# def format_plot_box(id, args, kwargs):
#     """Format data from a plot_box call."""
#     # Typically handles multiple data series for box plots
#     if len(args) >= 1:
#         data = args[0]
#         
#         # If data is a simple array or list
#         if isinstance(data, (np.ndarray, list)) and not isinstance(data[0], (list, np.ndarray)):
#             return pd.DataFrame({f"{id}_plot_box_values": data})
#             
#         # If data is a list of arrays (multiple box plots)
#         elif isinstance(data, (list, tuple)) and all(isinstance(x, (list, np.ndarray)) for x in data):
#             result = pd.DataFrame()
#             for i, values in enumerate(data):
#                 result[f"{id}_plot_box_group{i:02d}"] = pd.Series(values)
#             return result
#             
#         # If data is a dictionary
#         elif isinstance(data, dict):
#             result = pd.DataFrame()
#             for label, values in data.items():
#                 result[f"{id}_plot_box_{label}"] = pd.Series(values)
#             return result
#             
#     return pd.DataFrame()
# 
# def format_plot_conf_mat(id, args, kwargs):
#     """Format data from a plot_conf_mat call."""
#     # Placeholder implementation
#     if len(args) >= 1:
#         conf_mat = args[0]
#         
#         if hasattr(conf_mat, 'shape') and len(conf_mat.shape) == 2:
#             rows, cols = conf_mat.shape
#             row_indices, col_indices = np.meshgrid(range(rows), range(cols), indexing='ij')
#             
#             df = pd.DataFrame({
#                 f"{id}_conf_mat_row": row_indices.flatten(),
#                 f"{id}_conf_mat_col": col_indices.flatten(),
#                 f"{id}_conf_mat_value": conf_mat.flatten()
#             })
#             return df
#     return pd.DataFrame()
# 
# def format_plot_ecdf(id, args, kwargs):
#     """Format data from a plot_ecdf call."""
#     df = args
#     return df
# 
# def format_plot_fillv(id, args, kwargs):
#     """Format data from a plot_fillv call."""
#     starts, ends = args
#     df = pd.DataFrame(
#         {
#             f"{id}_plot_fillv_starts": starts,
#             f"{id}_plot_fillv_ends": ends,
#         }
#     )
#     return df
# 
# def format_plot_heatmap(id, args, kwargs):
#     """Format data from a plot_heatmap call."""
#     # Placeholder implementation
#     if len(args) >= 1:
#         data = args[0]
#         
#         if hasattr(data, 'shape') and len(data.shape) == 2:
#             rows, cols = data.shape
#             row_indices, col_indices = np.meshgrid(range(rows), range(cols), indexing='ij')
#             
#             df = pd.DataFrame({
#                 f"{id}_heatmap_row": row_indices.flatten(),
#                 f"{id}_heatmap_col": col_indices.flatten(),
#                 f"{id}_heatmap_value": data.flatten()
#             })
#             return df
#     return pd.DataFrame()
# 
# def format_plot_image(id, args, kwargs):
#     """Format data from a plot_image call."""
#     # Similar to imshow but may have different argument structure
#     if len(args) >= 1:
#         img = args[0]
#         
#         # Handle 2D grayscale images
#         if isinstance(img, np.ndarray) and img.ndim == 2:
#             rows, cols = img.shape
#             row_indices, col_indices = np.meshgrid(range(rows), range(cols), indexing='ij')
#             
#             df = pd.DataFrame({
#                 f"{id}_image_row": row_indices.flatten(),
#                 f"{id}_image_col": col_indices.flatten(),
#                 f"{id}_image_intensity": img.flatten()
#             })
#             return df
#             
#         # Handle RGB/RGBA images
#         elif isinstance(img, np.ndarray) and img.ndim == 3:
#             rows, cols, channels = img.shape
#             row_indices, col_indices = np.meshgrid(range(rows), range(cols), indexing='ij')
#             
#             data = {
#                 f"{id}_image_row": row_indices.flatten(),
#                 f"{id}_image_col": col_indices.flatten(),
#             }
#             
#             # Add channel data
#             channel_names = ['r', 'g', 'b', 'a']
#             for c in range(min(channels, len(channel_names))):
#                 data[f"{id}_image_{channel_names[c]}"] = img[:,:,c].flatten()
#                 
#             return pd.DataFrame(data)
#     
#     return pd.DataFrame()
# 
# def format_plot_joyplot(id, args, kwargs):
#     """Format data from a plot_joyplot call."""
#     # Joyplots (aka ridgeline plots) are typically a series of density distributions
#     if len(args) >= 1:
#         data = args[0]
#         
#         # Handle DataFrame input
#         if isinstance(data, pd.DataFrame):
#             # Make a copy to avoid modifying original
#             result = data.copy()
#             # Add prefix to column names if ID is provided
#             if id is not None:
#                 result.columns = [f"{id}_joyplot_{col}" for col in result.columns]
#             return result
#             
#         # Handle dictionary of arrays
#         elif isinstance(data, dict):
#             result = pd.DataFrame()
#             for group, values in data.items():
#                 result[f"{id}_joyplot_{group}"] = pd.Series(values)
#             return result
#             
#         # Handle list of arrays
#         elif isinstance(data, (list, tuple)) and all(isinstance(x, (np.ndarray, list)) for x in data):
#             result = pd.DataFrame()
#             for i, values in enumerate(data):
#                 result[f"{id}_joyplot_group{i:02d}"] = pd.Series(values)
#             return result
#             
#     return pd.DataFrame()
# 
# def format_plot_kde(id, args, kwargs):
#     """Format data from a plot_kde call."""
#     df = args
#     if id is not None:
#         df.columns = [f"{id}_plot_kde_{col}" for col in df.columns]
#     return df
# 
# def format_plot_line(id, args, kwargs):
#     """Format data from a plot_line call."""
#     # Similar to standard plot function
#     if len(args) >= 2:
#         x, y = args[:2]
#         
#         # Handle cases where y is multi-dimensional
#         if hasattr(y, 'ndim') and y.ndim == 2:
#             try:
#                 # Try using mngs.pd.force_df if available
#                 try:
#                     import mngs.pd
#                     data = {f"{id}_plot_line_x": x}
#                     for i in range(y.shape[1]):
#                         data[f"{id}_plot_line_y{i:02d}"] = y[:, i]
#                     return mngs.pd.force_df(data)
#                 except (ImportError, AttributeError):
#                     # Manual approach with individual columns
#                     df = pd.DataFrame({f"{id}_plot_line_x": x})
#                     for i in range(y.shape[1]):
#                         df[f"{id}_plot_line_y{i:02d}"] = y[:, i]
#                     return df
#             except Exception as e:
#                 import warnings
#                 warnings.warn(f"Error formatting plot_line data: {str(e)}")
#                 return pd.DataFrame()
#         
#         # Handle 1D arrays
#         elif isinstance(y, (np.ndarray, list, xr.DataArray)):
#             df = pd.DataFrame({
#                 f"{id}_plot_line_x": x,
#                 f"{id}_plot_line_y": y
#             })
#             return df
#             
#     return pd.DataFrame()
# 
# def format_plot_mean_ci(id, args, kwargs):
#     """Format data from a plot_mean_ci call."""
#     # Plot with mean and confidence intervals
#     if len(args) >= 3:
#         x, means, ci_low, ci_high = args[0], args[1], args[2], args[3] if len(args) > 3 else None
#         
#         data = {
#             f"{id}_mean_ci_x": x,
#             f"{id}_mean_ci_mean": means,
#             f"{id}_mean_ci_ci_low": ci_low,
#         }
#         
#         if ci_high is not None:
#             data[f"{id}_mean_ci_ci_high"] = ci_high
#         else:
#             # If only one CI value is provided, assume it's symmetric
#             data[f"{id}_mean_ci_ci_high"] = means + (means - ci_low)
#             
#         return pd.DataFrame(data)
#     return pd.DataFrame()
# 
# def format_plot_mean_std(id, args, kwargs):
#     """Format data from a plot_mean_std call."""
#     # Plot with mean and standard deviation
#     if len(args) >= 3:
#         x, means, stds = args[0], args[1], args[2]
#         
#         df = pd.DataFrame({
#             f"{id}_mean_std_x": x,
#             f"{id}_mean_std_mean": means,
#             f"{id}_mean_std_std": stds
#         })
#         return df
#     return pd.DataFrame()
# 
# def format_plot_median_iqr(id, args, kwargs):
#     """Format data from a plot_median_iqr call."""
#     # Plot with median and interquartile range
#     if len(args) >= 3:
#         x, medians, q1, q3 = args[0], args[1], args[2], args[3] if len(args) > 3 else None
#         
#         data = {
#             f"{id}_median_iqr_x": x,
#             f"{id}_median_iqr_median": medians,
#             f"{id}_median_iqr_q1": q1,
#         }
#         
#         if q3 is not None:
#             data[f"{id}_median_iqr_q3"] = q3
#         else:
#             # If only q1 is provided, assume it's symmetric
#             data[f"{id}_median_iqr_q3"] = medians + (medians - q1)
#             
#         return pd.DataFrame(data)
#     return pd.DataFrame()
# 
# def format_plot_raster(id, args, kwargs):
#     """Format data from a plot_raster call."""
#     df = args
#     return df
# 
# def format_plot_rectangle(id, args, kwargs):
#     """Format data from a plot_rectangle call."""
#     # Rectangles defined by [x, y, width, height]
#     if len(args) >= 4:
#         x, y, width, height = args[0], args[1], args[2], args[3]
#         
#         # Handle single rectangle
#         if all(isinstance(val, (int, float)) for val in [x, y, width, height]):
#             return pd.DataFrame({
#                 f"{id}_rectangle_x": [x],
#                 f"{id}_rectangle_y": [y],
#                 f"{id}_rectangle_width": [width],
#                 f"{id}_rectangle_height": [height]
#             })
#         
#         # Handle multiple rectangles (arrays)
#         elif all(isinstance(val, (np.ndarray, list)) for val in [x, y, width, height]):
#             return pd.DataFrame({
#                 f"{id}_rectangle_x": x,
#                 f"{id}_rectangle_y": y,
#                 f"{id}_rectangle_width": width,
#                 f"{id}_rectangle_height": height
#             })
#     
#     return pd.DataFrame()
# 
# def format_plot_scatter_hist(id, args, kwargs):
#     """Format data from a plot_scatter_hist call."""
#     # Typically contains scatter data and marginal histograms
#     if len(args) >= 2:
#         x, y = args[0], args[1]
#         
#         df = pd.DataFrame({
#             f"{id}_scatter_hist_x": x,
#             f"{id}_scatter_hist_y": y
#         })
#         
#         # If additional arguments for histogram bins are provided
#         if len(args) > 2 and isinstance(args[2], (np.ndarray, list)):
#             df[f"{id}_scatter_hist_x_bins"] = args[2]
#             
#         if len(args) > 3 and isinstance(args[3], (np.ndarray, list)):
#             df[f"{id}_scatter_hist_y_bins"] = args[3]
#             
#         return df
#     return pd.DataFrame()
# 
# def format_plot_shaded_line(id, args, kwargs):
#     """Format data from a plot_shaded_line call."""
#     # Typically a line plot with shaded confidence interval or error region
#     if len(args) >= 3:
#         x, y, shade_min, shade_max = args[0], args[1], args[2], args[3] if len(args) > 3 else None
#         
#         data = {
#             f"{id}_shaded_line_x": x,
#             f"{id}_shaded_line_y": y,
#             f"{id}_shaded_line_lower": shade_min
#         }
#         
#         if shade_max is not None:
#             data[f"{id}_shaded_line_upper"] = shade_max
#         elif shade_min is not None:
#             # If only shade_min is provided, assume it's symmetric around y
#             data[f"{id}_shaded_line_upper"] = y + (y - shade_min)
#             
#         return pd.DataFrame(data)
#     return pd.DataFrame()
# 
# def format_plot_violin(id, args, kwargs):
#     """Format data from a plot_violin call."""
#     # Custom implementation of violin plots may have different structure than matplotlib's
#     if len(args) >= 1:
#         data = args[0]
#         
#         # If data is a simple array or list
#         if isinstance(data, (np.ndarray, list)) and not isinstance(data[0], (list, np.ndarray)):
#             return pd.DataFrame({f"{id}_plot_violin_values": data})
#             
#         # If data is a list of arrays (multiple violin plots)
#         elif isinstance(data, (list, tuple)) and all(isinstance(x, (list, np.ndarray)) for x in data):
#             result = pd.DataFrame()
#             for i, values in enumerate(data):
#                 result[f"{id}_plot_violin_group{i:02d}"] = pd.Series(values)
#             return result
#             
#         # If data is a dictionary
#         elif isinstance(data, dict):
#             result = pd.DataFrame()
#             for label, values in data.items():
#                 result[f"{id}_plot_violin_{label}"] = pd.Series(values)
#             return result
#             
#         # If data is a DataFrame
#         elif isinstance(data, pd.DataFrame):
#             result = data.copy()
#             if id is not None:
#                 result.columns = [f"{id}_plot_violin_{col}" for col in result.columns]
#             return result
#             
#     return pd.DataFrame()
# 
# # ----------------------------------------
# # Seaborn function formatters
# # ----------------------------------------
# 
# def format_sns_barplot(id, args, kwargs):
#     """Format data from a sns_barplot call."""
#     df = args
#     # When xyhue, without errorbar
#     df = pd.DataFrame(
#         pd.Series(np.array(df).diagonal(), index=df.columns)
#     ).T
#     return df
# 
# def format_sns_boxplot(id, args, kwargs):
#     """Format data from a sns_boxplot call."""
#     df = args
#     if id is not None:
#         df.columns = [f"{id}_sns_boxplot_{col}" for col in df.columns]
#     return df
# 
# def format_sns_heatmap(id, args, kwargs):
#     """Format data from a sns_heatmap call."""
#     df = args
#     return df
# 
# def format_sns_histplot(id, args, kwargs):
#     """Format data from a sns_histplot call."""
#     df = args
#     return df
# 
# def format_sns_jointplot(id, args, kwargs):
#     """Format data from a sns_jointplot call."""
#     # Joint distribution plot in seaborn
#     if len(args) >= 1:
#         data = args[0]
#         x_var = kwargs.get('x')
#         y_var = kwargs.get('y')
#         
#         # Handle DataFrame input
#         if isinstance(data, pd.DataFrame) and x_var and y_var:
#             # Extract the relevant columns
#             x_data = data[x_var]
#             y_data = data[y_var]
#             
#             result = pd.DataFrame({
#                 f"{id}_joint_{x_var}": x_data,
#                 f"{id}_joint_{y_var}": y_data
#             })
#             return result
#             
#         # Handle direct x, y data arrays
#         elif isinstance(data, pd.DataFrame):
#             # If no x, y specified, return the whole dataframe
#             result = data.copy()
#             if id is not None:
#                 result.columns = [f"{id}_joint_{col}" for col in result.columns]
#             return result
#             
#         # Handle numpy arrays directly
#         elif all(arg in args for arg in range(2)) and isinstance(args[0], (np.ndarray, list)) and isinstance(args[1], (np.ndarray, list)):
#             x_data, y_data = args[0], args[1]
#             return pd.DataFrame({
#                 f"{id}_joint_x": x_data,
#                 f"{id}_joint_y": y_data
#             })
#             
#     return pd.DataFrame()
# 
# def format_sns_kdeplot(id, args, kwargs):
#     """Format data from a sns_kdeplot call."""
#     # Kernel density estimate plot
#     if len(args) >= 1:
#         data = args[0]
#         x_var = kwargs.get('x')
#         y_var = kwargs.get('y')
#         
#         # Handle DataFrame input with x, y variables
#         if isinstance(data, pd.DataFrame) and x_var:
#             if y_var:  # Bivariate KDE
#                 result = pd.DataFrame({
#                     f"{id}_kde_{x_var}": data[x_var],
#                     f"{id}_kde_{y_var}": data[y_var]
#                 })
#             else:  # Univariate KDE
#                 result = pd.DataFrame({
#                     f"{id}_kde_{x_var}": data[x_var]
#                 })
#             return result
#         
#         # Handle direct data array input
#         elif isinstance(data, (np.ndarray, list)):
#             y_data = args[1] if len(args) > 1 and isinstance(args[1], (np.ndarray, list)) else None
#             
#             if y_data is not None:  # Bivariate KDE
#                 return pd.DataFrame({
#                     f"{id}_kde_x": data,
#                     f"{id}_kde_y": y_data
#                 })
#             else:  # Univariate KDE
#                 return pd.DataFrame({
#                     f"{id}_kde_x": data
#                 })
#         
#         # Handle DataFrame input without x, y specified
#         elif isinstance(data, pd.DataFrame):
#             result = data.copy()
#             if id is not None:
#                 result.columns = [f"{id}_kde_{col}" for col in result.columns]
#             return result
#             
#     return pd.DataFrame()
# 
# def format_sns_lineplot(id, args, kwargs):
#     """Format data from a sns_lineplot call."""
#     # Line plot with potential error bands from seaborn
#     if len(args) >= 1:
#         data = args[0]
#         x_var = kwargs.get('x')
#         y_var = kwargs.get('y')
#         
#         # Handle DataFrame input with x, y variables
#         if isinstance(data, pd.DataFrame) and x_var and y_var:
#             result = pd.DataFrame({
#                 f"{id}_line_{x_var}": data[x_var],
#                 f"{id}_line_{y_var}": data[y_var]
#             })
#             
#             # Add grouping variable if present
#             hue_var = kwargs.get('hue')
#             if hue_var and hue_var in data.columns:
#                 result[f"{id}_line_{hue_var}"] = data[hue_var]
#                 
#             return result
#             
#         # Handle direct x, y data arrays
#         elif len(args) > 1 and isinstance(args[0], (np.ndarray, list)) and isinstance(args[1], (np.ndarray, list)):
#             x_data, y_data = args[0], args[1]
#             return pd.DataFrame({
#                 f"{id}_line_x": x_data,
#                 f"{id}_line_y": y_data
#             })
#             
#         # Handle DataFrame input without x, y specified
#         elif isinstance(data, pd.DataFrame):
#             result = data.copy()
#             if id is not None:
#                 result.columns = [f"{id}_line_{col}" for col in result.columns]
#             return result
#             
#     return pd.DataFrame()
# 
# def format_sns_pairplot(id, args, kwargs):
#     """Format data from a sns_pairplot call."""
#     # Grid of plots showing pairwise relationships
#     if len(args) >= 1:
#         data = args[0]
#         
#         # Handle DataFrame input
#         if isinstance(data, pd.DataFrame):
#             # For pairplot, just return the full DataFrame since it uses all variables
#             result = data.copy()
#             if id is not None:
#                 result.columns = [f"{id}_pair_{col}" for col in result.columns]
#                 
#             # Add vars or hue columns if specified
#             vars_list = kwargs.get('vars')
#             if vars_list and all(var in data.columns for var in vars_list):
#                 # Keep only the specified columns
#                 result = pd.DataFrame({f"{id}_pair_{col}": data[col] for col in vars_list})
#                 
#             return result
#     
#     return pd.DataFrame()
# 
# def format_sns_scatterplot(id, args, kwargs):
#     """Format data from a sns_scatterplot call."""
#     # Scatter plot with seaborn styling
#     if len(args) >= 1:
#         data = args[0]
#         x_var = kwargs.get('x')
#         y_var = kwargs.get('y')
#         
#         # Handle DataFrame input with x, y variables
#         if isinstance(data, pd.DataFrame) and x_var and y_var:
#             result = pd.DataFrame({
#                 f"{id}_scatter_{x_var}": data[x_var],
#                 f"{id}_scatter_{y_var}": data[y_var]
#             })
#             
#             # Add grouping or size variables if present
#             for extra_var in ['hue', 'size', 'style']:
#                 var_name = kwargs.get(extra_var)
#                 if var_name and var_name in data.columns:
#                     result[f"{id}_scatter_{var_name}"] = data[var_name]
#                     
#             return result
#             
#         # Handle direct x, y data arrays
#         elif len(args) > 1 and isinstance(args[0], (np.ndarray, list)) and isinstance(args[1], (np.ndarray, list)):
#             x_data, y_data = args[0], args[1]
#             return pd.DataFrame({
#                 f"{id}_scatter_x": x_data,
#                 f"{id}_scatter_y": y_data
#             })
#             
#         # Handle DataFrame input without x, y specified
#         elif isinstance(data, pd.DataFrame):
#             result = data.copy()
#             if id is not None:
#                 result.columns = [f"{id}_scatter_{col}" for col in result.columns]
#             return result
#             
#     return pd.DataFrame()
# 
# def format_sns_stripplot(id, args, kwargs):
#     """Format data from a sns_stripplot call."""
#     # Strip plot (categorical scatter plot)
#     if len(args) >= 1:
#         data = args[0]
#         x_var = kwargs.get('x')
#         y_var = kwargs.get('y')
#         
#         # Handle DataFrame input with x and/or y variables
#         if isinstance(data, pd.DataFrame):
#             result = pd.DataFrame()
#             
#             # Add x variable if specified
#             if x_var and x_var in data.columns:
#                 result[f"{id}_strip_{x_var}"] = data[x_var]
#                 
#             # Add y variable if specified
#             if y_var and y_var in data.columns:
#                 result[f"{id}_strip_{y_var}"] = data[y_var]
#                 
#             # Add grouping variable if present
#             hue_var = kwargs.get('hue')
#             if hue_var and hue_var in data.columns:
#                 result[f"{id}_strip_{hue_var}"] = data[hue_var]
#                 
#             # If we've added columns, return the result
#             if not result.empty:
#                 return result
#                 
#             # If no columns were explicitly specified, return all columns
#             result = data.copy()
#             if id is not None:
#                 result.columns = [f"{id}_strip_{col}" for col in result.columns]
#             return result
#             
#     return pd.DataFrame()
# 
# def format_sns_swarmplot(id, args, kwargs):
#     """Format data from a sns_swarmplot call."""
#     # Swarm plot (non-overlapping categorical scatter plot)
#     if len(args) >= 1:
#         data = args[0]
#         x_var = kwargs.get('x')
#         y_var = kwargs.get('y')
#         
#         # Handle DataFrame input with x and/or y variables
#         if isinstance(data, pd.DataFrame):
#             result = pd.DataFrame()
#             
#             # Add x variable if specified
#             if x_var and x_var in data.columns:
#                 result[f"{id}_swarm_{x_var}"] = data[x_var]
#                 
#             # Add y variable if specified
#             if y_var and y_var in data.columns:
#                 result[f"{id}_swarm_{y_var}"] = data[y_var]
#                 
#             # Add grouping variable if present
#             hue_var = kwargs.get('hue')
#             if hue_var and hue_var in data.columns:
#                 result[f"{id}_swarm_{hue_var}"] = data[hue_var]
#                 
#             # If we've added columns, return the result
#             if not result.empty:
#                 return result
#                 
#             # If no columns were explicitly specified, return all columns
#             result = data.copy()
#             if id is not None:
#                 result.columns = [f"{id}_swarm_{col}" for col in result.columns]
#             return result
#             
#     return pd.DataFrame()
# 
# def format_sns_violinplot(id, args, kwargs):
#     """Format data from a sns_violinplot call."""
#     df = args
#     return df
# 
# 
>>>>>>> origin/main
# def main():
#     # Line
#     fig, ax = mngs.plt.subplots()
#     ax.plot([1, 2, 3], [4, 5, 6], id="plot1")
#     ax.plot([4, 5, 6], [1, 2, 3], id="plot2")
#     mngs.io.save(fig, "./plots.png")
#     mngs.io.save(ax.export_as_csv(), "./plots.csv")
#
#     # No tracking
#     fig, ax = mngs.plt.subplots(track=False)
#     ax.plot([1, 2, 3], [4, 5, 6], id="plot3")
#     ax.plot([4, 5, 6], [1, 2, 3], id="plot4")
#     mngs.io.save(fig, "./plots_wo_tracking.png")
#     mngs.io.save(ax.export_as_csv(), "./plots_wo_tracking.csv")
#
#     # Scatter
#     fig, ax = mngs.plt.subplots()
#     ax.scatter([1, 2, 3], [4, 5, 6], id="scatter1")
#     ax.scatter([4, 5, 6], [1, 2, 3], id="scatter2")
#     mngs.io.save(fig, "./scatters.png")
#     mngs.io.save(ax.export_as_csv(), "./scatters.csv")
#
#     # Box
#     fig, ax = mngs.plt.subplots()
#     ax.boxplot([1, 2, 3], id="boxplot1")
#     mngs.io.save(fig, "./boxplot1.png")
#     mngs.io.save(ax.export_as_csv(), "./boxplot1.csv")
#
#     # Bar
#     fig, ax = mngs.plt.subplots()
#     ax.bar(["A", "B", "C"], [4, 5, 6], id="bar1")
#     mngs.io.save(fig, "./bar1.png")
#     mngs.io.save(ax.export_as_csv(), "./bar1.csv")
<<<<<<< HEAD
#
#     # # Bar
#     # # fig, ax = mngs.plt.subplots()
#     # fig, ax = plt.subplots()
#     # ax.bar(["A", "B", "C"], [4, 5, 6], id="bar2")
#     # mngs.io.save(fig, "./bar2.png")
#     # mngs.io.save(ax.export_as_csv(), "./bar2.csv")
#
#     # print(ax.export_as_csv())
#     # #    plot1_plot_x  plot1_plot_y  plot2_plot_x  ...  boxplot1_boxplot_x  bar1_bar_x  bar1_bar_y
#     # # 0           1.0           4.0           4.0  ...                 1.0           A         4.0
#     # # 1           2.0           5.0           5.0  ...                 2.0           B         5.0
#     # # 2           3.0           6.0           6.0  ...                 3.0           C         6.0
#
#     # print(ax.export_as_csv().keys())  # plot3 and plot 4 are not tracked
#     # # [3 rows x 11 columns]
#     # # Index(['plot1_plot_x', 'plot1_plot_y', 'plot2_plot_x', 'plot2_plot_y',
#     # #        'scatter1_scatter_x', 'scatter1_scatter_y', 'scatter2_scatter_x',
#     # #        'scatter2_scatter_y', 'boxplot1_boxplot_x', 'bar1_bar_x', 'bar1_bar_y'],
#     # #       dtype='object')
#
#     # # If a path is passed, the sigmaplot-friendly dataframe is saved as a csv file.
#     # ax.export_as_csv("../for_sigmaplot.csv")
#     # # Saved to: ../for_sigmaplot.csv
#
#
# if __name__ == "__main__":
#     # # Argument Parser
#     # import argparse
#     # parser = argparse.ArgumentParser(description='')
#     # parser.add_argument('--var', '-v', type=int, default=1, help='')
#     # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
#     # args = parser.parse_args()
#
=======
# 
# if __name__ == "__main__":
>>>>>>> origin/main
#     # Main
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False, agg=True
#     )
#     main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)
#
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/_subplots/_export_as_csv.py
# --------------------------------------------------------------------------------
