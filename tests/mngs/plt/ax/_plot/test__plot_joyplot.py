#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 21:55:17 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/_plot/test__plot_joyplot.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/_plot/test__plot_joyplot.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib.pyplot as plt
import mngs.plt.ax._plot._plot_joyplot as plot_joyplot_module
import numpy as np
import pandas as pd
import pytest

# @patch("joypy.plot_joyplot")
# def test_plot_joyplot_basic_functionality(mock_plot_joyplot):
#     """Test basic functionality of plot_joyplot function."""
#     # Set up mock return value
#     mock_fig = MagicMock()
#     mock_axes = MagicMock()
#     mock_plot_joyplot.return_value = (mock_fig, mock_axes)

#     # Create test data
#     data = pd.DataFrame(
#         {
#             "A": np.random.normal(0, 1, 100),
#             "B": np.random.normal(1, 1, 100),
#             "C": np.random.normal(2, 1, 100),
#         }
#     )

#     # Create a matplotlib figure and axes
#     fig, ax = plt.subplots()

#     # Call the function
#     result_ax = plot_joyplot_module.plot_joyplot(ax, data)

#     # Verify mock was called with correct arguments
#     mock_plot_joyplot.assert_called_once()
#     args, kwargs = mock_plot_joyplot.call_args
#     assert "data" in kwargs
#     assert kwargs["data"] is data

#     # Verify function returns correct object
#     assert result_ax is ax

#     # Clean up
#     plt.close(fig)

# @patch("joypy.plot_joyplot")
# def test_plot_joyplot_with_vertical_orientation(mock_plot_joyplot):
#     """Test plot_joyplot function with vertical orientation."""
#     # Set up mock return value
#     mock_fig = MagicMock()
#     mock_axes = MagicMock()
#     mock_plot_joyplot.return_value = (mock_fig, mock_axes)

#     # Create test data
#     data = pd.DataFrame(
#         {"A": np.random.normal(0, 1, 100), "B": np.random.normal(1, 1, 100)}
#     )

#     # Create a matplotlib figure and axes
#     fig, ax = plt.subplots()

#     # Call the function with vertical orientation
#     result_ax = plot_joyplot_module.plot_joyplot(
#         ax, data, orientation="vertical"
#     )

#     # Verify mock was called with correct arguments
#     mock_plot_joyplot.assert_called_once()
#     args, kwargs = mock_plot_joyplot.call_args
#     assert "data" in kwargs
#     assert kwargs["data"] is data
#     assert kwargs["orientation"] == "vertical"

#     # Verify function returns correct object
#     assert result_ax is ax

#     # Clean up
#     plt.close(fig)

# @patch("joypy.plot_joyplot")
# def test_plot_joyplot_with_custom_parameters(mock_plot_joyplot):
#     """Test plot_joyplot function with custom parameters."""
#     # Set up mock return value
#     mock_fig = MagicMock()
#     mock_axes = MagicMock()
#     mock_plot_joyplot.return_value = (mock_fig, mock_axes)

#     # Create test data
#     data = pd.DataFrame(
#         {"A": np.random.normal(0, 1, 100), "B": np.random.normal(1, 1, 100)}
#     )

#     # Create a matplotlib figure and axes
#     fig, ax = plt.subplots()

#     # Custom parameters
#     custom_params = {
#         "colormap": plt.cm.viridis,
#         "title": "Custom Title",
#         "overlap": 0.5,
#         "figsize": (8, 6),
#     }

#     # Call the function with custom parameters
#     result_ax = plot_joyplot_module.plot_joyplot(ax, data, **custom_params)

#     # Verify mock was called with correct arguments
#     mock_plot_joyplot.assert_called_once()
#     args, kwargs = mock_plot_joyplot.call_args
#     assert "data" in kwargs
#     assert kwargs["data"] is data
#     for key, value in custom_params.items():
#         assert kwargs[key] == value

#     # Verify function returns correct object
#     assert result_ax is ax

#     # Clean up
#     plt.close(fig)


def test_plot_joyplot_basic():
    fig, ax = plt.subplots()
    data = pd.DataFrame(
        {"A": np.random.normal(0, 1, 100), "B": np.random.normal(1, 1, 100)}
    )

    ax = plot_joyplot_module.plot_joyplot(ax, data)

    # Saving
    from mngs.io import save

    spath = f"./basic.jpg"
    save(fig, spath)

    # Check saved file
    ACTUAL_SAVE_DIR = __file__.replace(".py", "_out")
    actual_spath = os.path.join(ACTUAL_SAVE_DIR, spath)
    assert os.path.exists(actual_spath), f"Failed to save figure to {spath}"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_plot/_plot_joyplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-02 09:03:23 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_plot/_plot_joyplot.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/ax/_plot/_plot_joyplot.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import warnings
# 
# import joypy
# 
# from .._style._set_xyt import set_xyt as mngs_plt_set_xyt
# 
# 
# def plot_joyplot(ax, data, orientation="vertical", **kwargs):
#     # FIXME; orientation should be handled
#     fig, axes = joypy.joyplot(
#         data=data,
#         **kwargs,
#     )
# 
#     if orientation == "vertical":
#         ax = mngs_plt_set_xyt(ax, None, "Density", "Joyplot")
#     elif orientation == "horizontal":
#         ax = mngs_plt_set_xyt(ax, "Density", None, "Joyplot")
#     else:
#         warnings.warn(
#             "orientation must be either of 'vertical' or 'horizontal'"
#         )
# 
#     return ax
# 
# 
# # def plot_vertical_joyplot(ax, data, **kwargs):
# #     return _plot_joyplot(ax, data, "vertical", **kwargs)
# 
# 
# # def plot_horizontal_joyplot(ax, data, **kwargs):
# #     return _plot_joyplot(ax, data, "horizontal", **kwargs)
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_plot/_plot_joyplot.py
# --------------------------------------------------------------------------------
