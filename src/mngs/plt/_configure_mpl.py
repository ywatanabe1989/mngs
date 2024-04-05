#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-05 17:07:26 (ywatanabe)"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def rgba_to_hex(rgba):
    return "#{:02x}{:02x}{:02x}{:02x}".format(
        int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3] * 255)
    )


def normalize_rgba(rgba):
    rgba = list(rgba)
    rgba[0] /= 255
    rgba[1] /= 255
    rgba[2] /= 255
    rgba = tuple(rgba)
    return rgba


def configure_mpl(
    plt,
    # Fig Size
    fig_size_mm=(160, 100),
    fig_scale=1.0,
    # DPI
    dpi_display=100,
    dpi_save=300,
    # Font Size
    font_size_base=8,
    font_size_title=8,
    font_size_axis_label=8,
    font_size_tick_label=7,
    font_size_legend=6,
    # Hide spines
    hide_top_right_spines=True,
    # line
    line_width=0.1,
    # Color transparency
    alpha=0.75,
    # Whether to print configurations or not
    verbose=False,
    **kwargs,
):
    """
    Configures Matplotlib and Seaborn settings for publication-quality plots.
    For axis control, refer to the mngs.plt.ax module.

    Example:
        plt, cc = configure_mpl(plt)

        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        for i_cc, cc_str in enumerate(cc):
            phase_shift = i_cc * np.pi / len(cc)
            y = np.sin(x + phase_shift)
            ax.plot(x, y, color=cc[cc_str], label=f"{cc_str}")
        ax.legend()
        plt.show()

    Parameters:
        plt (matplotlib.pyplot):
            Matplotlib pyplot module.

        fig_size_mm (tuple of int, optional):
            Figure width and height in millimeters. Defaults to (160, 100).

        fig_scale (float, optional):
            Scaling factor for the figure size. Defaults to 1.0.

        dpi_display (int, optional):
            Display resolution in dots per inch. Defaults to 100.

        dpi_save (int, optional):
            Resolution for saved figures in dots per inch. Defaults to 300.

        font_size_title (int, optional):
            Font size for titles. Defaults to 8.

        font_size_axis_label (int, optional):
            Font size for axis labels. Defaults to 8.

        font_size_tick_label (int, optional):
            Font size for tick labels. Defaults to 7.

        font_size_legend (int, optional):
            Font size for legend text. Defaults to 6.

        hide_top_right_spines (bool, optional):
            If True, hides the top and right spines of the plot. Defaults to True.

        alpha (float, optional):
            Color transparency. Defaults to 0.75.

        verbose (bool, optional):
            If True, prints the configuration settings. Defaults to True.

    Returns:
        dict: A dictionary of the custom colors used in the configuration.
    """

    COLORS_RGBA = {
        "blue": (0, 128, 192, alpha),
        "red": (255, 70, 50, alpha),
        "pink": (255, 150, 200, alpha),
        "green": (20, 180, 20, alpha),
        "yellow": (230, 160, 20, alpha),
        "grey": (128, 128, 128, alpha),
        "purple": (200, 50, 255, alpha),
        "lightblue": (20, 200, 200, alpha),
        "brown": (128, 0, 0, alpha),
        "darkblue": (0, 0, 100, alpha),
        "orange": (228, 94, 50, alpha),
        "white": (255, 255, 255, alpha),
        "black": (0, 0, 0, alpha),
    }
    COLORS_HEX = {k: rgba_to_hex(v) for k, v in COLORS_RGBA.items()}
    COLORS_RGBA_NORM = {c: normalize_rgba(v) for c, v in COLORS_RGBA.items()}

    # Normalize figure size from mm to inches
    figsize_inch = (fig_size_mm[0] / 25.4, fig_size_mm[1] / 25.4)

    # Update Matplotlib configuration
    plt.rcParams.update(
        {
            # Resolution
            "figure.dpi": dpi_display,
            "savefig.dpi": dpi_save,
            # Figure Size
            "figure.figsize": figsize_inch,
            # Font Size
            "font.size": font_size_base,
            # Title
            "axes.titlesize": font_size_title,
            # Axis
            "axes.labelsize": font_size_axis_label,
            # Ticks
            "xtick.labelsize": font_size_tick_label,
            "ytick.labelsize": font_size_tick_label,
            # Legend
            "legend.fontsize": font_size_legend,
            # Top and Right Axes
            "axes.spines.top": not hide_top_right_spines,
            "axes.spines.right": not hide_top_right_spines,
            # Custom color cycle
            "axes.prop_cycle": plt.cycler(color=COLORS_RGBA_NORM.values()),
            # Line
            "lines.linewidth": line_width,
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
        print(f"Font Size (Title): {font_size_title} pt")
        print(f"Font Size (X/Y Label): {font_size_axis_label} pt")
        print(f"Font Size (Tick Label): {font_size_tick_label} pt")
        print(f"Font Size (Legend): {font_size_legend} pt")
        print(f"Hide Top and Right Axes: {hide_top_right_spines}")
        print(f"Custom Colors (RGBA):")
        for color_str, rgba in COLORS_RGBA.items():
            print(f"  {color_str}: {rgba}")
        print("-" * 40)

    return plt, COLORS_RGBA_NORM


if __name__ == "__main__":
    plt, cc = configure_mpl(plt)

    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    for i_cc, cc_str in enumerate(cc):
        phase_shift = i_cc * np.pi / len(cc)
        y = np.sin(x + phase_shift)
        ax.plot(x, y, color=cc[cc_str], label=f"{cc_str}")
    ax.legend()
    plt.show()
