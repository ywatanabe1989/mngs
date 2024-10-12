#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-15 18:10:23 (ywatanabe)"

import matplotlib.pyplot as plt
import mngs
import numpy as np


def configure_mpl(
    plt,
    # Fig Size
    fig_size_mm=(160, 100),
    fig_scale=1.0,
    # DPI
    dpi_display=100,
    dpi_save=300,
    # # Font Size
    # font_size_base=10,
    # font_size_title=10,
    # font_size_axis_label=10,
    # font_size_tick_label=8,
    # font_size_legend=8,
    # Hide spines
    hide_top_right_spines=True,
    # line
    line_width=0.5,
    # Color transparency
    alpha=0.9,
    # Whether to print configurations or not
    verbose=False,
    **kwargs,
):
    """
    Configures Matplotlib and Seaborn settings for publication-quality plots.
    For axis control, refer to the mngs.plt.ax module.

    Example:
        plt, CC = configure_mpl(plt)

        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        for i_cc, cc_str in enumerate(cc):
            phase_shift = i_cc * np.pi / len(CC)
            y = np.sin(x + phase_shift)
            ax.plot(x, y, color=CC[cc_str], label=f"{cc_str}")
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

        # font_size_title (int, optional):
        #     Font size for titles. Defaults to 8.

        # font_size_axis_label (int, optional):
        #     Font size for axis labels. Defaults to 8.

        # font_size_tick_label (int, optional):
        #     Font size for tick labels. Defaults to 7.

        # font_size_legend (int, optional):
        #     Font size for legend text. Defaults to 6.

        hide_top_right_spines (bool, optional):
            If True, hides the top and right spines of the plot. Defaults to True.

        alpha (float, optional):
            Color transparency. Defaults to 0.75.

        verbose (bool, optional):
            If True, prints the configuration settings. Defaults to True.

    Returns:
        dict: A dictionary of the custom colors used in the configuration.
    """

    RGBA = {
        k: mngs.plt.update_alpha(v, alpha)
        for k, v in mngs.plt.PARAMS["RGBA"].items()
    }
    RGBA_NORM = {
        k: tuple(mngs.plt.update_alpha(v, alpha))
        for k, v in mngs.plt.PARAMS["RGBA_NORM"].items()
    }
    RGBA_NORM_FOR_CYCLE = {
        k: tuple(mngs.plt.update_alpha(v, alpha))
        for k, v in mngs.plt.PARAMS["RGBA_NORM_FOR_CYCLE"].items()
    }  # without black, gr'a'y and white

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
            # # Font Size
            # "font.size": font_size_base,
            # # Title
            # "axes.titlesize": font_size_title,
            # # Axis
            # "axes.labelsize": font_size_axis_label,
            # # Ticks
            # "xtick.labelsize": font_size_tick_label,
            # "ytick.labelsize": font_size_tick_label,
            # # Legend
            # "legend.fontsize": font_size_legend,
            # Top and Right Axes
            "axes.spines.top": not hide_top_right_spines,
            "axes.spines.right": not hide_top_right_spines,
            # Custom color cycle
            "axes.prop_cycle": plt.cycler(color=RGBA_NORM_FOR_CYCLE.values()),
            # "axes.prop_cycle": plt.cycler(
            #     color=[tuple(rgba) for rgba in RGBA.values()]
            # ),
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
        # print(f"Font Size (Title): {font_size_title} pt")
        # print(f"Font Size (X/Y Label): {font_size_axis_label} pt")
        # print(f"Font Size (Tick Label): {font_size_tick_label} pt")
        # print(f"Font Size (Legend): {font_size_legend} pt")
        print(f"Hide Top and Right Axes: {hide_top_right_spines}")
        print(f"Custom Colors (RGBA):")
        for color_str, rgba in RGBA.items():
            print(f"  {color_str}: {rgba}")
        print("-" * 40)

    return plt, RGBA_NORM


if __name__ == "__main__":
    plt, CC = configure_mpl(plt)

    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
    x = np.linspace(0, 10, 100)
    for i_cc, cc_str in enumerate(CC):
        phase_shift = i_cc * np.pi / len(CC)
        y = np.sin(x + phase_shift)
        axes[0].plot(x, y, label="Default color cycle")
        axes[1].plot(x, y, color=CC[cc_str], label=f"{cc_str}")
    axes[0].legend()
    axes[1].legend()
    plt.show()
