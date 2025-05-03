#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-02 17:42:18 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/utils/_configure_mpl.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/utils/_configure_mpl.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import mngs
import numpy as np


def configure_mpl(
    plt,
    fig_size_mm=(160, 100),
    fig_scale=1.0,
    dpi_display=100,
    dpi_save=300,
    fontsize="medium",
    autolayout=True,
    hide_top_right_spines=True,
    line_width=1.0,  # Increased from 0.5 for better visibility
    alpha=0.85,  # Adjusted for better contrast
    verbose=False,
    **kwargs,
) -> Tuple[Any, Dict]:
    """Configures Matplotlib settings for publication-quality plots.

    Parameters
    ----------
    plt : matplotlib.pyplot
        Matplotlib pyplot module
    fig_size_mm : tuple of int, optional
        Figure width and height in millimeters, by default (160, 100)
    fig_scale : float, optional
        Scaling factor for figure size, by default 1.0
    dpi_display : int, optional
        Display resolution in DPI, by default 100
    dpi_save : int, optional
        Saving resolution in DPI, by default 300
    fontsize : Union[str, int, float], optional
        Base font size ('xx-small' to 'xx-large' or points), by default 'medium'
        Other sizes are derived from this:
        - Title: 125% of base
        - Labels: 100% of base
        - Ticks/Legend: 85% of base
    autolayout : bool, optional
        Whether to enable automatic tight layout, by default True
    hide_top_right_spines : bool, optional
        Whether to hide top and right spines, by default True
    line_width : float, optional
        Default line width, by default 1.0
    alpha : float, optional
        Color transparency, by default 0.85
    verbose : bool, optional
        Whether to print configuration details, by default False

    Returns
    -------
    tuple
        (plt, dict of RGBA colors)
    """
    # Convert base font size
    base_size = _convert_font_size(fontsize)

    # Ensure minimum sizes for different elements with better proportions
    title_size = max(base_size * 1.25, 10.0)  # Increased for better hierarchy
    label_size = max(base_size * 1.0, 9.0)  # Minimum 9pt for good readability
    small_size = max(
        base_size * 0.85, 8.0
    )  # Increased ratio for better legibility

    # Colors
    RGBA = {
        k: mngs.plt.color.update_alpha(v, alpha)
        for k, v in mngs.plt.color.PARAMS["RGBA"].items()
    }

    RGBA_NORM = {
        k: tuple(mngs.plt.color.update_alpha(v, alpha))
        for k, v in mngs.plt.color.PARAMS["RGBA_NORM"].items()
    }

    RGBA_NORM_FOR_CYCLE = {
        k: tuple(mngs.plt.color.update_alpha(v, alpha))
        for k, v in mngs.plt.color.PARAMS["RGBA_NORM_FOR_CYCLE"].items()
    }

    # Normalize figure size from mm to inches
    figsize_inch = (
        fig_size_mm[0] / 25.4 * fig_scale,
        fig_size_mm[1] / 25.4 * fig_scale,
    )

    # Update Matplotlib configuration
    plt.rcParams.update(
        {
            # Resolution
            "figure.dpi": dpi_display,
            "savefig.dpi": dpi_save,
            # Figure Size
            "figure.figsize": figsize_inch,
            # Font Sizes
            "font.size": base_size,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "xtick.labelsize": small_size,
            "ytick.labelsize": small_size,
            "legend.fontsize": small_size,
            # Auto Layout
            "figure.autolayout": autolayout,
            # Top and Right Axes
            "axes.spines.top": not hide_top_right_spines,
            "axes.spines.right": not hide_top_right_spines,
            # Spine width
            "axes.linewidth": 0.8,  # Slightly thicker axes lines
            # Custom color cycle
            "axes.prop_cycle": plt.cycler(
                color=list(RGBA_NORM_FOR_CYCLE.values())
            ),
            # Line
            "lines.linewidth": line_width,
            "lines.markersize": 6.0,  # Better default marker size
            # Grid (if used)
            "grid.linewidth": 0.6,
            "grid.alpha": 0.3,
        }
    )

    if verbose:
        print("\n" + "-" * 40)
        print("Matplotlib has been configured as follows:\n")
        print(f"Figure DPI (Display): {dpi_display} DPI")
        print(f"Figure DPI (Save): {dpi_save} DPI")
        print(
            f"Figure Size (Not the Axis Size): "
            f"{fig_size_mm[0] * fig_scale:.1f} x "
            f"{fig_size_mm[1] * fig_scale:.1f} mm (width x height)"
        )
        print("\nFont Sizes:")
        print(f"  Base Size: {base_size:.1f}pt")
        print(f"  Title: {title_size:.1f}pt (125% of base, min 10pt)")
        print(f"  Axis Labels: {label_size:.1f}pt (100% of base, min 9pt)")
        print(f"  Tick Labels: {small_size:.1f}pt (85% of base, min 8pt)")
        print(f"  Legend: {small_size:.1f}pt (85% of base, min 8pt)")
        print(f"\nHide Top and Right Axes: {hide_top_right_spines}")
        print(f"Line Width: {line_width}")
        print(f"\nCustom Colors (RGBA):")
        for color_str, rgba in RGBA.items():
            print(f"  {color_str}: {rgba}")
        print("-" * 40)

    return plt, RGBA_NORM


def _convert_font_size(size: Union[str, int, float]) -> float:
    """Converts various font size specifications to numerical values.

    Parameters
    ----------
    size : Union[str, int, float]
        Font size specification. Can be:
        - String: 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
        - Numeric: direct point size value

    Returns
    -------
    float
        Font size in points
    """
    if isinstance(size, str):
        size_map = {
            "xx-small": 9,
            "x-small": 11,
            "small": 13,
            "medium": 15,
            "large": 18,
            "x-large": 22,
            "xx-large": 26,
        }
        return size_map.get(size.lower(), 15)
    elif isinstance(size, (int, float)):
        return max(float(size), 9.0)  # Ensure minimum size of 9
    else:
        raise ValueError(f"Unsupported font size type: {type(size)}")


if __name__ == "__main__":
    plt, CC = configure_mpl(plt)
    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
    x = np.linspace(0, 10, 100)
    for idx_cc, cc_str in enumerate(CC):
        phase_shift = idx_cc * np.pi / len(CC)
        y = np.sin(x + phase_shift)
        axes[0].plot(x, y, label="Default color cycle")
        axes[1].plot(x, y, color=CC[cc_str], label=f"{cc_str}")
    axes[0].legend()
    axes[1].legend()
    plt.show()

# EOF