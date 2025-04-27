#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 16:38:34 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__configure_mpl.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__configure_mpl.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/_configure_mpl.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-17 13:59:03 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/plt/_configure_mpl.py
#
# __file__ = "./src/mngs/plt/_configure_mpl.py"
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

import sys

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.plt._configure_mpl import _convert_font_size, configure_mpl


def test_convert_font_size_str():
    assert _convert_font_size("small") == 10
    assert _convert_font_size("XX-LARGE") == 18


def test_convert_font_size_numeric():
    assert _convert_font_size(12) == 12.0


def test_configure_mpl_returns_cycle():
    new_plt, cc = configure_mpl(plt, fontsize=8, hide_top_right_spines=False)
    assert new_plt is plt
    assert isinstance(cc, dict)
    assert all(isinstance(v, tuple) for v in cc.values())

# EOF