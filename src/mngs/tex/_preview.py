#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "ywatanabe (2024-11-02 23:58:12)"
# File: ./mngs_repo/src/mngs/tex/_preview.py

import numpy as np


def preview(tex_str_list):
    r"""
    Generate a preview of LaTeX strings.

    Example
    -------
    tex_strings = ["x^2", "\sum_{i=1}^n i"]
    fig = preview(tex_strings)
    mngs.plt.show()

    Parameters
    ----------
    tex_str_list : list of str
        List of LaTeX strings to preview

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the previews
    """
    from ..plt import subplots
    fig, axes = subplots(nrows=len(tex_str_list), ncols=1, figsize=(10, 3*len(tex_str_list)))
    axes = np.atleast_1d(axes)
    for ax, tex_string in zip(axes, tex_str_list):
        ax.text(0.5, 0.7, tex_string, size=20, ha='center', va='center')
        ax.text(0.5, 0.3, f"${tex_string}$", size=20, ha='center', va='center')
        ax.hide_spines()
    fig.tight_layout()
    return fig


# EOF
