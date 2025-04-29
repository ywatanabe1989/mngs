#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 14:11:26 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/test__joyplot.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/test__joyplot.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import mngs.plt.ax._joyplot as joyplot_module
import numpy as np
import pandas as pd
import pytest


@patch("joypy.joyplot")
def test_joyplot_basic_functionality(mock_joyplot):
    """Test basic functionality of joyplot function."""
    # Set up mock return value
    mock_fig = MagicMock()
    mock_axes = MagicMock()
    mock_joyplot.return_value = (mock_fig, mock_axes)

    # Create test data
    data = pd.DataFrame(
        {
            "A": np.random.normal(0, 1, 100),
            "B": np.random.normal(1, 1, 100),
            "C": np.random.normal(2, 1, 100),
        }
    )

    # Create a matplotlib figure and axes
    fig, ax = plt.subplots()

    # Call the function
    result_ax = joyplot_module.joyplot(ax, data)

    # Verify mock was called with correct arguments
    mock_joyplot.assert_called_once()
    args, kwargs = mock_joyplot.call_args
    assert "data" in kwargs
    assert kwargs["data"] is data

    # Verify function returns correct object
    assert result_ax is ax

    # Clean up
    plt.close(fig)


@patch("joypy.joyplot")
def test_joyplot_with_vertical_orientation(mock_joyplot):
    """Test joyplot function with vertical orientation."""
    # Set up mock return value
    mock_fig = MagicMock()
    mock_axes = MagicMock()
    mock_joyplot.return_value = (mock_fig, mock_axes)

    # Create test data
    data = pd.DataFrame(
        {"A": np.random.normal(0, 1, 100), "B": np.random.normal(1, 1, 100)}
    )

    # Create a matplotlib figure and axes
    fig, ax = plt.subplots()

    # Call the function with vertical orientation
    result_ax = joyplot_module.joyplot(ax, data, orientation="vertical")

    # Verify mock was called with correct arguments
    mock_joyplot.assert_called_once()
    args, kwargs = mock_joyplot.call_args
    assert "data" in kwargs
    assert kwargs["data"] is data
    assert kwargs["orientation"] == "vertical"

    # Verify function returns correct object
    assert result_ax is ax

    # Clean up
    plt.close(fig)


@patch("joypy.joyplot")
def test_joyplot_with_custom_parameters(mock_joyplot):
    """Test joyplot function with custom parameters."""
    # Set up mock return value
    mock_fig = MagicMock()
    mock_axes = MagicMock()
    mock_joyplot.return_value = (mock_fig, mock_axes)

    # Create test data
    data = pd.DataFrame(
        {"A": np.random.normal(0, 1, 100), "B": np.random.normal(1, 1, 100)}
    )

    # Create a matplotlib figure and axes
    fig, ax = plt.subplots()

    # Custom parameters
    custom_params = {
        "colormap": plt.cm.viridis,
        "title": "Custom Title",
        "overlap": 0.5,
        "figsize": (8, 6),
    }

    # Call the function with custom parameters
    result_ax = joyplot_module.joyplot(ax, data, **custom_params)

    # Verify mock was called with correct arguments
    mock_joyplot.assert_called_once()
    args, kwargs = mock_joyplot.call_args
    assert "data" in kwargs
    assert kwargs["data"] is data
    for key, value in custom_params.items():
        assert kwargs[key] == value

    # Verify function returns correct object
    assert result_ax is ax

    # Clean up
    plt.close(fig)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_joyplot.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-06 00:04:26 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/plt/ax/_joyplot.py
# 
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-08-26 08:38:55 (ywatanabe)"
# # ./src/mngs/plt/ax/_joyplot.py
# 
# import joypy
# import mngs
# from ._set_xyt import set_xyt
# 
# # def plot_joy(df, cols_NT):
# #     df_plot = df[cols_NT]
# #     fig, axes = joypy.joyplot(
# #         data=df_plot,
# #         colormap=plt.cm.viridis,
# #         title="Distribution of Ranked Data",
# #         labels=cols_NT,
# #         overlap=0.5,
# #         orientation="vertical",
# #     )
# #     plt.xlabel("Variables")
# #     plt.ylabel("Rank")
# #     return fig
# 
# 
# def joyplot(ax, data, **kwargs):
#     fig, axes = joypy.joyplot(
#         data=data,
#         **kwargs,
#     )
# 
#     if kwargs.get("orientation") == "vertical":
#         xlabel = None
#         ylabel = "Density"
#     else:
#         xlabel = "Density"
#         ylabel = None
# 
#     ax = set_xyt(ax, xlabel, ylabel, "Joyplot")
# 
#     return ax
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_joyplot.py
# --------------------------------------------------------------------------------
