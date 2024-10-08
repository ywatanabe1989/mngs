#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-02 23:35:05 (ywatanabe)"
# /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/tex/_preview.py

import mngs
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
    fig, axes = mngs.plt.subplots(nrows=len(tex_str_list), ncols=1, figsize=(10, 3*len(tex_str_list)))
    axes = np.atleast_1d(axes)
    for ax, tex_string in zip(axes, tex_str_list):
        ax.text(0.5, 0.7, tex_string, size=20, ha='center', va='center')
        ax.text(0.5, 0.3, f"${tex_string}$", size=20, ha='center', va='center')
        ax.hide_spines()
    fig.tight_layout()
    return fig
