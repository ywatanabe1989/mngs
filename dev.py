#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-29 16:38:04 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/dev.py
# ----------------------------------------
import os
__FILE__ = (
    "./dev.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import matplotlib.pyplot as plt_mpl
import numpy as np
from matplotlib_venn import venn2

import mngs
import mngs.io
import mngs.plt as plt_mngs

matplotlib.use("agg")

plt_mpl, CC = mngs.plt.configure_mpl(plt_mpl)


def prepare_pairs():
    kwargs = {"ncols": 2}

    # Matplotlib
    fig_mpl, axes_mpl = plt_mpl.subplots(**kwargs)
    axis_mpl = axes_mpl[0]

    # mngs
    fig_mngs, axes_mngs = plt_mngs.subplots(**kwargs)
    axis_mngs = axes_mngs[0]

    return {
        "plt": (plt_mngs, plt_mpl),
        "fig": (fig_mngs, fig_mpl),
        "axes": (axes_mngs, axes_mpl),
        "axis": (axis_mngs, axis_mpl),
    }


def print_axes_types(pairs):
    print(type(pairs["axes"][0]), type(pairs["axes"][1]))
    print(type(pairs["axis"][0]), type(pairs["axis"][1]))


def plot_attribute_venn_diagrams(
    pairs, save_dir="/tmp/mngs_plt_comparison", max_attrs_to_list=64
):
    """Plots Venn diagrams showing attribute differences and adds mngs-specific attributes as text."""
    print(f"\n--- Plotting Attribute Venn Diagrams to {save_dir} ---")

    for label, (mngs_obj, mpl_obj) in pairs.items():
        mngs_name = type(mngs_obj).__name__
        mpl_name = type(mpl_obj).__name__
        print(
            f"    Generating Venn diagram for: {label} ({mngs_name} vs {mpl_name})"
        )

        fig_venn = None

        # Get attribute sets using dir()
        mngs_attrs = set(dir(mngs_obj))
        mpl_attrs = set(dir(mpl_obj))

        # Consider only exposed attributes
        mpl_attrs_exposed = {
            attr for attr in mpl_attrs if not attr.startswith("_")
        }
        mngs_attrs_exposed = {
            attr for attr in mngs_attrs if not attr.startswith("_")
        }

        # mngs-only attributes
        mngs_only_attrs_exposed = sorted(
            list(mngs_attrs_exposed - mpl_attrs_exposed)
        )
        # Union attributes
        union_attrs_exposed = sorted(
            list(mngs_attrs_exposed & mpl_attrs_exposed)
        )

        # Create a new figure for each Venn diagram using mngs.plt
        fig_venn_mngs, ax_venn_mngs = plt_mngs.subplots(figsize=(10, 10))

        # Create the Venn diagram on the axis
        v = venn2(
            [mngs_attrs_exposed, mpl_attrs_exposed],
            set_labels=(
                f"mngs\n({mngs_name})",
                f"mpl\n({mpl_name})",
            ),
            ax=ax_venn_mngs,
            set_colors=(CC["blue"], CC["red"]),
            alpha=0.5,
        )

        # Add text for mngs-only attributes in the left side of the figure
        y_start = 0.9
        y_step = 0.03
        font_size = 8

        # Add a header for the attribute list
        ax_venn_mngs.text(
            0.05,
            0.95,
            f"MNGS attributes:",
            fontsize=font_size + 2,
            weight="bold",
            transform=ax_venn_mngs.transAxes,
        )

        # List attributes with smaller step size
        for ii, attr in enumerate(
            mngs_only_attrs_exposed + union_attrs_exposed
        ):
            is_mngs_only = attr in mngs_only_attrs_exposed
            union_color = [(b + r) / 2 for b, r in zip(CC["blue"], CC["red"])]

            c = CC["blue"] if is_mngs_only else union_color
            ax_venn_mngs.text(
                0.05,
                y_start - ii * y_step,
                attr,
                fontsize=font_size,
                transform=ax_venn_mngs.transAxes,
                c=c,
            )

        # Set title using mngs wrapper
        ax_venn_mngs.set_xyt(
            tt=f"Attribute Comparison: {label}\n({mngs_name} vs {mpl_name})"
        )
        # Adjust layout slightly
        fig_venn_mngs.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the figure using mngs.io
        save_path = os.path.join(save_dir, f"{label}_venn.png")
        # fig_venn_mngs.savefig(save_path)
        mngs.io.save(fig_venn_mngs, save_path)
        print(f"    Saved Venn diagram to: {save_path}")

        # Save as csv
        save_path = os.path.join(save_dir, f"{label}_attributes.csv")
        df = mngs.pd.force_df(
            {
                "mngs": mngs_attrs_exposed,
                "union": union_attrs_exposed,
                "matplotlib": mpl_attrs_exposed,
            }
        )
        mngs.io.save(df, save_path)


def plot_and_save_figures(pairs, save_dir="/tmp/mngs_plt_comparison"):
    """Adds simple plots to the axes and saves the figures."""
    print(f"\n--- Plotting and Saving Figures to {save_dir} ---")

    # Get mngs axis and figure
    axis_mngs = pairs["axis"][0]
    fig_mngs = pairs["fig"][0]

    # Get mpl axis and figure
    axis_mpl = pairs["axis"][1]
    fig_mpl = pairs["fig"][1]

    # Define some data
    x_data = np.linspace(0, 10, 50)
    y_data_1 = np.sin(x_data)
    y_data_2 = np.cos(x_data)

    try:
        # Plot on mngs axis (using its methods)
        # Use id for potential tracking in mngs
        axis_mngs.plot_(
            x_data, yy=y_data_1, label="sin(x) - mngs", id="mngs_sin"
        )
        # Access the second axis in the mngs AxesWrapper

        # axis_mngs.plot(
        #     x_data, y_data_2, label="cos(x) - mngs", id="mngs_cos", color="red"
        # )
        # Add titles via FigWrapper
        fig_mngs.supxyt(t="MNGS Plot Comparison")
        # Add legends
        axis_mngs.legend()
        # Save using mngs.io (which handles FigWrapper)
        mngs_save_path = os.path.join(save_dir, "lineplot_mngs.png")
        mngs.io.save(fig_mngs, mngs_save_path)
        # fig_mngs.savefig(mngs_save_path)
        print(f"  Saved mngs figure to: {mngs_save_path}")

    except Exception as e_mngs:
        print(f"  Error plotting/saving mngs figure: {e_mngs}")

    try:
        # Plot on matplotlib axis (using its methods)
        axis_mpl.plot(x_data, y_data_1, label="sin(x) - mpl")
        # Access the second axis in the mpl ndarray
        pairs["axes"][1][1].plot(
            x_data, y_data_2, label="cos(x) - mpl", color="red"
        )
        # Add titles via mpl Figure
        fig_mpl.suptitle("Matplotlib Plot Comparison")
        # Add legends
        for ax_mpl_single in pairs["axes"][1]:
            ax_mpl_single.legend()
        # Save using matplotlib figure's method
        mpl_save_path = os.path.join(save_dir, "lineplot_matplotlib.png")
        fig_mpl.savefig(mpl_save_path)
        print(f"  Saved matplotlib figure to: {mpl_save_path}")

    except Exception as e_mpl:
        print(f"  Error plotting/saving matplotlib figure: {e_mpl}")


def close_figures(pairs):
    """Closes the matplotlib figures."""
    print("\n--- Closing Figures ---")
    plt_mpl.close()
    plt_mngs.close()


def main():
    """Main function to run the comparison."""
    print("--- Comparing mngs.plt wrappers with matplotlib counterparts ---")
    pairs = prepare_pairs()
    print_axes_types(pairs)
    # print_attribute_differences(pairs)
    plot_attribute_venn_diagrams(pairs)
    plot_and_save_figures(pairs)
    close_figures(pairs)
    print("\n--- Comparison finished ---")


if __name__ == "__main__":
    main()

fig, axes = mngs.plt.subplots(ncols=2)
dir(fig)
dir(axes)

# EOF