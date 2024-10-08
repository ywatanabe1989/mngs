#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-04 17:24:03 (ywatanabe)"
# /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/plt/ax/_half_violin.py

"""
1. Functionality:
   - (e.g., Executes XYZ operation)
2. Input:
   - (e.g., Required data for XYZ)
3. Output:
   - (e.g., Results of XYZ operation)
4. Prerequisites:
   - (e.g., Necessary dependencies for XYZ)

(Remove me: Please fill docstrings above, while keeping the bulette point style, and remove this instruction line)
"""


import seaborn as sns
import matplotlib.pyplot as plt

def half_violin(ax, data=None, x=None, y=None, hue=None, **kwargs):
    # Prepare data
    df = data.copy()
    if hue is None:
        df['_hue'] = 'default'
        hue = '_hue'

    # Add fake hue for the right side
    df['_fake_hue'] = df[hue] + '_right'

    # Adjust hue_order and palette if provided
    if 'hue_order' in kwargs:
        kwargs['hue_order'] = kwargs['hue_order'] + [h + '_right' for h in kwargs['hue_order']]

    if 'palette' in kwargs:
        palette = kwargs['palette']
        if isinstance(palette, dict):
            kwargs['palette'] = {**palette, **{k + '_right': v for k, v in palette.items()}}
        elif isinstance(palette, list):
            kwargs['palette'] = palette + palette

    # Plot
    sns.violinplot(data=df, x=x, y=y, hue='_fake_hue', split=True, ax=ax, **kwargs)

    # Remove right half of violins
    for collection in ax.collections:
        if isinstance(collection, plt.matplotlib.collections.PolyCollection):
            collection.set_clip_path(None)

    # Adjust legend
    if ax.legend_ is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(handles)//2], labels[:len(labels)//2])

    return ax
