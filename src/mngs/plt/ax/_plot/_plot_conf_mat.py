#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-02 09:20:23 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_plot/_plot_conf_mat.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/ax/_plot/_plot_conf_mat.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ...utils._calc_bacc_from_conf_mat import calc_bacc_from_conf_mat
from .._style._extend import extend as mngs_plt_extend


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
    calc_bacc: bool = False,
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
    calc_bacc : bool, optional
        Calculate Balanced Accuracy from Confusion Matrix

    Returns
    -------
    Union[plt.Axes, Tuple[plt.Axes, float]]
        Axes object and optionally balanced accuracy

    Example
    -------
    >>> data = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])
    >>> fig, ax = plt.subplots()
    >>> ax, bacc = plot_conf_mat(ax, data, x_labels=['A','B','C'],
    ...                     y_labels=['X','Y','Z'], calc_bacc=True)
    >>> print(f"Balanced Accuracy: {bacc:.3f}")
    Balanced Accuracy: 0.889
    """

    assert isinstance(
        axis, matplotlib.axes._axes.Axes
    ), "First argument must be a matplotlib axis"

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    bacc_val = calc_bacc_from_conf_mat(data.values)
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

    if calc_bacc:
        return axis, bacc_val
    else:
        return axis, None

# EOF