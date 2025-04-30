#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 20:33:52 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_plot_violin_half.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/ax/_plot_violin_half.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def plot_half_violin(ax, data=None, x=None, y=None, hue=None, **kwargs):

    assert isinstance(
        ax, matplotlib.axes._axes.Axes
    ), "First argument must be a matplotlib axis"

    # Prepare data
    df = data.copy()
    if hue is None:
        df["_hue"] = "default"
        hue = "_hue"

    # Add fake hue for the right side
    df["_fake_hue"] = df[hue] + "_right"

    # Adjust hue_order and palette if provided
    if "hue_order" in kwargs:
        kwargs["hue_order"] = kwargs["hue_order"] + [
            h + "_right" for h in kwargs["hue_order"]
        ]

    if "palette" in kwargs:
        palette = kwargs["palette"]
        if isinstance(palette, dict):
            kwargs["palette"] = {
                **palette,
                **{k + "_right": v for k, v in palette.items()},
            }
        elif isinstance(palette, list):
            kwargs["palette"] = palette + palette

    # Plot
    sns.violinplot(
        data=df, x=x, y=y, hue="_fake_hue", split=True, ax=ax, **kwargs
    )

    # Remove right half of violins
    for collection in ax.collections:
        if isinstance(collection, plt.matplotlib.collections.PolyCollection):
            collection.set_clip_path(None)

    # Adjust legend
    if ax.legend_ is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[: len(handles) // 2], labels[: len(labels) // 2])

    return ax

# EOF