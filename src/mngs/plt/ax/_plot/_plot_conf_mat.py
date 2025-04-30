#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 08:49:21 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_plot/_plot_conf_mat.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/ax/_plot/_plot_conf_mat.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import sys
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ....context import suppress_output
from .._adjust._extend import extend as mngs_plt_extend


def plot_conf_mat(
    axis: plt.Axes,
    data: Union[np.ndarray, pd.DataFrame],
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    cbar: bool = True,
    cbar_kw: dict = {},
    label_rotation_xy: Tuple[float, float] = (15, 15),
    x_extend_ratio: float = 1.0,
    y_extend_ratio: float = 1.0,
    bacc: bool = False,
    **kwargs,
) -> Union[plt.Axes, Tuple[plt.Axes, float]]:
    """Creates a confusion matrix heatmap with optional balanced accuracy.

    Parameters
    ----------
    axis : plt.Axes
        Matplotlib axes to plot on
    data : Union[np.ndarray, pd.DataFrame]
        Confusion matrix data
    x_labels : Optional[List[str]], optional
        Labels for predicted classes
    y_labels : Optional[List[str]], optional
        Labels for true classes
    title : str, optional
        Plot title
    cmap : str, optional
        Colormap name
    cbar : bool, optional
        Whether to show colorbar
    cbar_kw : dict, optional
        Colorbar parameters
    label_rotation_xy : Tuple[float, float], optional
        (x,y) label rotation angles
    x_extend_ratio : float, optional
        X-axis extension ratio
    y_extend_ratio : float, optional
        Y-axis extension ratio
    bacc : bool, optional
        Return balanced accuracy

    Returns
    -------
    Union[plt.Axes, Tuple[plt.Axes, float]]
        Axes object and optionally balanced accuracy

    Example
    -------
    >>> data = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])
    >>> fig, ax = plt.subplots()
    >>> ax, bacc = plot_conf_mat(ax, data, x_labels=['A','B','C'],
    ...                     y_labels=['X','Y','Z'], bacc=True)
    >>> print(f"Balanced Accuracy: {bacc:.3f}")
    Balanced Accuracy: 0.889
    """

    assert isinstance(
        axis, matplotlib.axes._axes.Axes
    ), "First argument must be a matplotlib axis"

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    bacc_val = calc_bACC_from_cm(data.values)
    title = f"{title} (bACC = {bacc_val:.3f})"

    res = sns.heatmap(
        data,
        ax=axis,
        cmap=cmap,
        annot=True,
        fmt=",d",
        cbar=False,
        vmin=0,
        **kwargs,
    )

    res.invert_yaxis()

    for _, spine in res.spines.items():
        spine.set_visible(False)

    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)

    if x_labels is not None:
        axis.set_xticklabels(x_labels)
    if y_labels is not None:
        axis.set_yticklabels(y_labels)

    axis = mngs_plt_extend(axis, x_extend_ratio, y_extend_ratio)
    if data.shape[0] == data.shape[1]:
        axis.set_box_aspect(1)
        axis.set_xticklabels(
            axis.get_xticklabels(),
            rotation=label_rotation_xy[0],
            fontdict={"verticalalignment": "top"},
        )
        axis.set_yticklabels(
            axis.get_yticklabels(),
            rotation=label_rotation_xy[1],
            fontdict={"horizontalalignment": "right"},
        )

    if bacc:
        return axis, bacc_val
    return axis


def calc_bACC_from_cm(confusion_matrix: np.ndarray) -> float:
    """Calculates balanced accuracy from confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix array

    Returns
    -------
    float
        Balanced accuracy score

    Example
    -------
    >>> cm = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])
    >>> bacc = calc_bACC_from_cm(cm)
    >>> print(f"Balanced Accuracy: {bacc:.3f}")
    Balanced Accuracy: 0.889
    """
    with suppress_output():
        try:
            per_class = np.diag(confusion_matrix) / np.nansum(
                confusion_matrix, axis=1
            )
            bacc = np.nanmean(per_class)
        except:
            bacc = np.nan
        return round(bacc, 3)


def main() -> None:
    import mngs

    data = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])
    fig, ax = mngs.plt.subplots()
    ax, bacc = plot_conf_mat(
        ax, data, x_labels=["A", "B", "C"], y_labels=["X", "Y", "Z"], bacc=True
    )
    print(bacc)
    mngs.io.save(fig, "fig.jpg")


if __name__ == "__main__":
    import matplotlib
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# def plot_conf_mat(
#     ax,
#     data,
#     x_labels=None,
#     y_labels=None,
#     title="Confusion Matrix",
#     cmap="Blues",
#     cbar=True,
#     cbar_kw={},
#     label_rotation_xy=(15, 15),
#     x_extend_ratio=1.0,
#     y_extend_ratio=1.0,
#     bacc=False,
#     **kwargs,
# ):
#     """
#     Create a confusion matrix heatmap from a numpy array or pandas DataFrame.

#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes
#         The axes object to plot on
#     data : array-like or pandas.DataFrame
#         The confusion matrix data
#     x_labels : list, optional
#         Labels for x-axis
#     y_labels : list, optional
#         Labels for y-axis
#     title : str, optional
#         Title of the plot
#     cmap : str, optional
#         Colormap for the heatmap
#     cbar : bool, optional
#         Whether to display a colorbar
#     cbar_kw : dict, optional
#         Additional arguments for colorbar
#     label_rotation_xy : tuple, optional
#         Rotation angles for x and y labels
#     x_extend_ratio : float, optional
#         Ratio to extend x-axis
#     y_extend_ratio : float, optional
#         Ratio to extend y-axis
#     bacc : bool, optional
#         Whether to return the balanced accuracy
#     **kwargs : dict
#         Additional arguments for sns.heatmap

#     Returns
#     -------
#     ax : matplotlib.axes.Axes
#         The axes object with the confusion matrix plotted
#     bacc : float, optional
#         The balanced accuracy (if return_bacc is True)

#     Example
#     -------
#     data = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])
#     fig, ax = plt.subplots()
#     ax = plot_conf_mat(ax, data, x_labels=['A', 'B', 'C'], y_labels=['X', 'Y', 'Z'])
#     plt.show()
#     """

#     fig = ax.fig

#     if not isinstance(data, pd.DataFrame):
#         data = pd.DataFrame(data)

#     bacc_val = calc_bACC_from_cm(data.values)
#     title = f"{title} (bACC = {bacc_val:.3f})"

#     res = sns.heatmap(
#         data,
#         ax=ax,
#         cmap=cmap,
#         annot=True,
#         fmt=",d",
#         cbar=False,
#         vmin=0,
#         **kwargs,
#     )

#     res.invert_yaxis()

#     for _, spine in res.spines.items():
#         spine.set_visible(False)

#     ax.set_xlabel("Predicted label")
#     ax.set_ylabel("True label")
#     ax.set_title(title)

#     if x_labels is not None:
#         ax.set_xticklabels(x_labels)
#     if y_labels is not None:
#         ax.set_yticklabels(y_labels)

#     ax = extend(ax, x_extend_ratio, y_extend_ratio)
#     if data.shape[0] == data.shape[1]:
#         ax.set_box_aspect(1)
#         ax.set_xticklabels(
#             ax.get_xticklabels(),
#             rotation=label_rotation_xy[0],
#             fontdict={"verticalalignment": "top"},
#         )
#         ax.set_yticklabels(
#             ax.get_yticklabels(),
#             rotation=label_rotation_xy[1],
#             fontdict={"horizontalalignment": "right"},
#         )

#     # if cbar:
#     #     divider = make_axes_locatable(ax)
#     #     cax = divider.append_axes("right", size="5%", pad=0.1)
#     #     bbox = ax.get_position()
#     #     width_orig = bbox.x1 - bbox.x0
#     #     width_tgt = width_orig * x_extend_ratio
#     #     dx = width_orig - width_tgt
#     #     cax = mngs.plt.ax.shift(cax, dx=-dx * 2.54, dy=0)
#     #     fig.add_axes(cax)

#     #     vmax = data.values.max().astype(int)
#     #     norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
#     #     cbar = fig.colorbar(
#     #         plt.cm.ScalarMappable(norm=norm, cmap=cmap),
#     #         cax=cax,
#     #         **cbar_kw
#     #     )
#     #     cbar.locator = matplotlib.ticker.MaxNLocator(nbins=4)
#     #     cbar.update_ticks()
#     #     cbar.outline.set_edgecolor("white")

#     if bacc:
#         return ax, bacc_val
#     return ax

# def calc_bACC_from_cm(confusion_matrix):
#     """
#     Calculate balanced accuracy from a confusion matrix.

#     Parameters
#     ----------
#     confusion_matrix : array-like
#         The confusion matrix

#     Returns
#     -------
#     float
#         The balanced accuracy

#     Example
#     -------
#     cm = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])
#     bacc = calc_bACC_from_cm(cm)
#     print(f"Balanced Accuracy: {bacc:.3f}")
#     """
#     with suppress_output():
#         try:
#             per_class = np.diag(confusion_matrix) / np.nansum(
#                 confusion_matrix, axis=1
#             )
#             bacc = np.nanmean(per_class)
#         except:
#             bacc = np.nan
#         return round(bacc, 3)

# def main():
#     data = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])
#     fig, ax = mngs.plt.subplots()
#     ax, bacc = plot_conf_mat(
#         ax, data, x_labels=["A", "B", "C"], y_labels=["X", "Y", "Z"], bacc=True
#     )
#     print(bacc)
#     mngs.io.save(fig, "fig.jpg")

# if __name__ == "__main__":
#     import mngs

#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False, agg=True
#     )
#     main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)

#

# EOF