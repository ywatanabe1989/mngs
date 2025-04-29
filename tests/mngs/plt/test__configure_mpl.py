#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:15:01 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__configure_mpl.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__configure_mpl.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib.pyplot as plt
import pytest
from mngs.plt._configure_mpl import _convert_font_size, configure_mpl


def test_convert_font_size_string():
    """Tests font size conversion from string values."""
    assert _convert_font_size("small") == 10
    assert _convert_font_size("medium") == 12
    assert _convert_font_size("large") == 14
    assert _convert_font_size("xx-large") == 18
    assert _convert_font_size("unknown") == 12  # Default for unknown strings


def test_convert_font_size_numeric():
    """Tests font size conversion from numeric values."""
    assert _convert_font_size(10) == 10.0
    assert _convert_font_size(14.5) == 14.5
    assert isinstance(_convert_font_size(10), float)


def test_convert_font_size_invalid():
    """Tests error handling for invalid font size types."""
    with pytest.raises(ValueError):
        _convert_font_size(None)

    with pytest.raises(ValueError):
        _convert_font_size([10])


def test_configure_mpl_basic():
    """Tests basic configuration of matplotlib."""
    original_rcparams = plt.rcParams.copy()

    try:
        # Run with default parameters
        plt_result, colors = configure_mpl(plt)

        # Check that plt was returned correctly
        assert plt_result is plt

        # Check that colors dictionary is not empty
        assert isinstance(colors, dict)
        assert len(colors) > 0

        # Check that rcParams were updated
        assert plt.rcParams["figure.dpi"] == 100
        assert plt.rcParams["savefig.dpi"] == 300
        assert plt.rcParams["font.size"] == 12.0  # Default 'medium'
    finally:
        # Restore original settings
        plt.rcParams.update(original_rcparams)


def test_configure_mpl_custom_params():
    """Tests configuration with custom parameters."""
    original_rcparams = plt.rcParams.copy()

    try:
        # Run with custom parameters
        plt_result, colors = configure_mpl(
            plt,
            fig_size_mm=(200, 150),
            dpi_display=150,
            dpi_save=600,
            fontsize="large",
            hide_top_right_spines=False,
            line_width=1.0,
            alpha=0.7,
        )

        # Check that custom parameters were applied
        assert plt.rcParams["figure.dpi"] == 150
        assert plt.rcParams["savefig.dpi"] == 600
        assert plt.rcParams["font.size"] == 14.0  # 'large'
        assert plt.rcParams["lines.linewidth"] == 1.0
        assert plt.rcParams["axes.spines.top"] == True  # Not hidden
        assert plt.rcParams["axes.spines.right"] == True  # Not hidden

        # Check figure size conversion (mm to inches)
        assert plt.rcParams["figure.figsize"][0] == pytest.approx(200 / 25.4)
        assert plt.rcParams["figure.figsize"][1] == pytest.approx(150 / 25.4)
    finally:
        # Restore original settings
        plt.rcParams.update(original_rcparams)


def test_configure_mpl_font_scaling():
    """Tests correct scaling of font sizes relative to base size."""
    original_rcparams = plt.rcParams.copy()

    try:
        # Use a specific base size
        plt_result, colors = configure_mpl(plt, fontsize=10)

        # Check that derived font sizes are correctly scaled
        assert plt.rcParams["font.size"] == 10.0  # Base size
        assert plt.rcParams["axes.titlesize"] == 12.0  # 120% of base
        assert plt.rcParams["axes.labelsize"] == 10.0  # 100% of base
        assert plt.rcParams["xtick.labelsize"] == 8.0  # 80% of base
        assert plt.rcParams["ytick.labelsize"] == 8.0  # 80% of base
        assert plt.rcParams["legend.fontsize"] == 8.0  # 80% of base
    finally:
        # Restore original settings
        plt.rcParams.update(original_rcparams)


def test_configure_mpl_verbose_output(capsys):
    """Tests that verbose mode prints configuration details."""
    original_rcparams = plt.rcParams.copy()

    try:
        # Run with verbose output
        plt_result, colors = configure_mpl(plt, verbose=True)

        # Capture printed output
        captured = capsys.readouterr()

        # Verify key information is in the output
        assert "Matplotlib has been configured as follows" in captured.out
        assert "Figure DPI (Display): 100 DPI" in captured.out
        assert "Figure DPI (Save): 300 DPI" in captured.out
        assert "Base Size: 12.0pt" in captured.out
        assert "Hide Top and Right Axes: True" in captured.out
        assert "Custom Colors (RGBA)" in captured.out
    finally:
        # Restore original settings
        plt.rcParams.update(original_rcparams)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_configure_mpl.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-17 13:59:03 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/plt/_configure_mpl.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_configure_mpl.py"
# 
# from typing import Union
# 
# import matplotlib.pyplot as plt
# import mngs
# import numpy as np
# 
# 
# def _convert_font_size(size: Union[str, int, float]) -> float:
#     """Converts various font size specifications to numerical values.
# 
#     Parameters
#     ----------
#     size : Union[str, int, float]
#         Font size specification. Can be:
#         - String: 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
#         - Numeric: direct point size value
# 
#     Returns
#     -------
#     float
#         Font size in points
#     """
# 
#     if isinstance(size, str):
#         size_map = {
#             "xx-small": 6,
#             "x-small": 8,
#             "small": 10,
#             "medium": 12,
#             "large": 14,
#             "x-large": 16,
#             "xx-large": 18,
#         }
#         return size_map.get(size.lower(), 12)
#     elif isinstance(size, (int, float)):
#         return float(size)
#     else:
#         raise ValueError(f"Unsupported font size type: {type(size)}")
# 
# 
# def configure_mpl(
#     plt,
#     fig_size_mm=(160, 100),
#     fig_scale=1.0,
#     dpi_display=100,
#     dpi_save=300,
#     fontsize="medium",
#     autolayout=True,
#     hide_top_right_spines=True,
#     line_width=0.5,
#     alpha=0.9,
#     verbose=False,
#     **kwargs,
# ):
#     """Configures Matplotlib settings for publication-quality plots.
# 
#     Parameters
#     ----------
#     plt : matplotlib.pyplot
#         Matplotlib pyplot module
#     fig_size_mm : tuple of int, optional
#         Figure width and height in millimeters, by default (160, 100)
#     fig_scale : float, optional
#         Scaling factor for figure size, by default 1.0
#     dpi_display : int, optional
#         Display resolution in DPI, by default 100
#     dpi_save : int, optional
#         Saving resolution in DPI, by default 300
#     fontsize : Union[str, int, float], optional
#         Base font size ('xx-small' to 'xx-large' or points), by default 'medium'
#         Other sizes are derived from this:
#         - Title: 120% of base
#         - Labels: 100% of base
#         - Ticks/Legend: 80% of base
#     auto_layout : bool, optional
#         Whether to enable automatic tight layout, by default True
#     hide_top_right_spines : bool, optional
#         Whether to hide top and right spines, by default True
#     line_width : float, optional
#         Default line width, by default 0.5
#     alpha : float, optional
#         Color transparency, by default 0.9
#     verbose : bool, optional
#         Whether to print configuration details, by default False
# 
#     Returns
#     -------
#     tuple
#         (plt, dict of RGBA colors)
#     """
# 
#     # Convert base font size
#     base_size = _convert_font_size(fontsize)
# 
#     # Colors
#     RGBA = {
#         k: mngs.plt.update_alpha(v, alpha)
#         for k, v in mngs.plt.PARAMS["RGBA"].items()
#     }
#     RGBA_NORM = {
#         k: tuple(mngs.plt.update_alpha(v, alpha))
#         for k, v in mngs.plt.PARAMS["RGBA_NORM"].items()
#     }
#     RGBA_NORM_FOR_CYCLE = {
#         k: tuple(mngs.plt.update_alpha(v, alpha))
#         for k, v in mngs.plt.PARAMS["RGBA_NORM_FOR_CYCLE"].items()
#     }
# 
#     # Normalize figure size from mm to inches
#     figsize_inch = (fig_size_mm[0] / 25.4, fig_size_mm[1] / 25.4)
# 
#     # Update Matplotlib configuration
#     plt.rcParams.update(
#         {
#             # Resolution
#             "figure.dpi": dpi_display,
#             "savefig.dpi": dpi_save,
#             # Figure Size
#             "figure.figsize": figsize_inch,
#             # Font Sizes (all relative to base_size)
#             "font.size": base_size,
#             "axes.titlesize": base_size * 1.2,
#             "axes.labelsize": base_size * 1.0,
#             "xtick.labelsize": base_size * 0.8,
#             "ytick.labelsize": base_size * 0.8,
#             "legend.fontsize": base_size * 0.8,
#             # Auto Layout
#             "figure.autolayout": autolayout,
#             # Top and Right Axes
#             "axes.spines.top": not hide_top_right_spines,
#             "axes.spines.right": not hide_top_right_spines,
#             # Custom color cycle
#             "axes.prop_cycle": plt.cycler(color=RGBA_NORM_FOR_CYCLE.values()),
#             # Line
#             "lines.linewidth": line_width,
#         }
#     )
# 
# 
#     if verbose:
#         print("\n" + "-" * 40)
#         print("Matplotlib has been configured as follows:\n")
#         print(f"Figure DPI (Display): {dpi_display} DPI")
#         print(f"Figure DPI (Save): {dpi_save} DPI")
#         print(
#             f"Figure Size (Not the Axis Size): "
#             f"{fig_size_mm[0] * fig_scale:.1f} x "
#             f"{fig_size_mm[1] * fig_scale:.1f} mm (width x height)"
#         )
#         print("\nFont Sizes:")
#         print(f"  Base Size: {base_size:.1f}pt")
#         print(f"  Title: {base_size * 1.2:.1f}pt (120% of base)")
#         print(f"  Axis Labels: {base_size * 1.0:.1f}pt (100% of base)")
#         print(f"  Tick Labels: {base_size * 0.8:.1f}pt (80% of base)")
#         print(f"  Legend: {base_size * 0.8:.1f}pt (80% of base)")
#         print(f"\nHide Top and Right Axes: {hide_top_right_spines}")
#         print(f"Line Width: {line_width}")
#         print(f"\nCustom Colors (RGBA):")
#         for color_str, rgba in RGBA.items():
#             print(f"  {color_str}: {rgba}")
#         print("-" * 40)
# 
#     # if verbose:
#     #     print("\n" + "-" * 40)
#     #     print("Matplotlib has been configured as follows:\n")
#     #     print(f"Figure DPI (Display): {dpi_display} DPI")
#     #     print(f"Figure DPI (Save): {dpi_save} DPI")
#     #     print(
#     #         f"Figure Size (Not the Axis Size): "
#     #         f"{fig_size_mm[0] * fig_scale:.1f} x "
#     #         f"{fig_size_mm[1] * fig_scale:.1f} mm (width x height)"
#     #     )
#     #     # print(f"Font Size (Title): {font_size_title} pt")
#     #     # print(f"Font Size (X/Y Label): {font_size_axis_label} pt")
#     #     # print(f"Font Size (Tick Label): {font_size_tick_label} pt")
#     #     # print(f"Font Size (Legend): {font_size_legend} pt")
#     #     print(f"Hide Top and Right Axes: {hide_top_right_spines}")
#     #     print(f"Custom Colors (RGBA):")
#     #     for color_str, rgba in RGBA.items():
#     #         print(f"  {color_str}: {rgba}")
#     #     print("-" * 40)
# 
#     return plt, RGBA_NORM
# 
# 
# if __name__ == "__main__":
#     plt, CC = configure_mpl(plt)
# 
#     fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
#     x = np.linspace(0, 10, 100)
#     for i_cc, cc_str in enumerate(CC):
#         phase_shift = i_cc * np.pi / len(CC)
#         y = np.sin(x + phase_shift)
#         axes[0].plot(x, y, label="Default color cycle")
#         axes[1].plot(x, y, color=CC[cc_str], label=f"{cc_str}")
#     axes[0].legend()
#     axes[1].legend()
#     plt.show()
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_configure_mpl.py
# --------------------------------------------------------------------------------
