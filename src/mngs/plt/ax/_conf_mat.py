#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-28 11:02:59 (ywatanabe)"
# /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/plt/ax/_conf_mat.py

import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mngs

def conf_mat(
    ax,
    data,
    x_labels=None,
    y_labels=None,
    title="Confusion Matrix",
    cmap="Blues",
    cbar=True,
    cbar_kw={},
    label_rotation_xy=(15, 15),
    x_extend_ratio=1.0,
    y_extend_ratio=1.0,
    bacc=False,
    **kwargs
):
    """
    Create a confusion matrix heatmap from a numpy array or pandas DataFrame.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on
    data : array-like or pandas.DataFrame
        The confusion matrix data
    x_labels : list, optional
        Labels for x-axis
    y_labels : list, optional
        Labels for y-axis
    title : str, optional
        Title of the plot
    cmap : str, optional
        Colormap for the heatmap
    cbar : bool, optional
        Whether to display a colorbar
    cbar_kw : dict, optional
        Additional arguments for colorbar
    label_rotation_xy : tuple, optional
        Rotation angles for x and y labels
    x_extend_ratio : float, optional
        Ratio to extend x-axis
    y_extend_ratio : float, optional
        Ratio to extend y-axis
    bacc : bool, optional
        Whether to return the balanced accuracy
    **kwargs : dict
        Additional arguments for sns.heatmap

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the confusion matrix plotted
    bacc : float, optional
        The balanced accuracy (if return_bacc is True)

    Example
    -------
    data = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])
    fig, ax = plt.subplots()
    ax = conf_mat(ax, data, x_labels=['A', 'B', 'C'], y_labels=['X', 'Y', 'Z'])
    plt.show()
    """

    fig = ax.fig

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    bacc_val = calc_bACC_from_cm(data.values)
    title = f"{title} (bACC = {bacc_val:.3f})"

    res = sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=",d",
        cbar=False,
        vmin=0,
        **kwargs
    )

    res.invert_yaxis()

    for _, spine in res.spines.items():
        spine.set_visible(False)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    if x_labels is not None:
        ax.set_xticklabels(x_labels)
    if y_labels is not None:
        ax.set_yticklabels(y_labels)

    ax = mngs.plt.ax.extend(ax, x_extend_ratio, y_extend_ratio)
    if data.shape[0] == data.shape[1]:
        ax.set_box_aspect(1)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=label_rotation_xy[0],
            fontdict={"verticalalignment": "top"},
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=label_rotation_xy[1],
            fontdict={"horizontalalignment": "right"},
        )

    # if cbar:
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.1)
    #     bbox = ax.get_position()
    #     width_orig = bbox.x1 - bbox.x0
    #     width_tgt = width_orig * x_extend_ratio
    #     dx = width_orig - width_tgt
    #     cax = mngs.plt.ax.shift(cax, dx=-dx * 2.54, dy=0)
    #     fig.add_axes(cax)

    #     vmax = data.values.max().astype(int)
    #     norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    #     cbar = fig.colorbar(
    #         plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    #         cax=cax,
    #         **cbar_kw
    #     )
    #     cbar.locator = matplotlib.ticker.MaxNLocator(nbins=4)
    #     cbar.update_ticks()
    #     cbar.outline.set_edgecolor("white")

    if bacc:
        return ax, bacc_val
    return ax

def calc_bACC_from_cm(confusion_matrix):
    """
    Calculate balanced accuracy from a confusion matrix.

    Parameters
    ----------
    confusion_matrix : array-like
        The confusion matrix

    Returns
    -------
    float
        The balanced accuracy

    Example
    -------
    cm = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])
    bacc = calc_bACC_from_cm(cm)
    print(f"Balanced Accuracy: {bacc:.3f}")
    """
    with mngs.gen.suppress_output():
        try:
            per_class = np.diag(confusion_matrix) / np.nansum(confusion_matrix, axis=1)
            bacc = np.nanmean(per_class)
        except:
            bacc = np.nan
        return round(bacc, 3)

def main():
    data = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])
    fig, ax = mngs.plt.subplots()
    ax, bacc = conf_mat(ax, data, x_labels=['A', 'B', 'C'], y_labels=['X', 'Y', 'Z'], bacc=True)
    print(bacc)
    mngs.io.save(fig, "fig.jpg")

if __name__ == '__main__':
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt, verbose=False, agg=True)
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
