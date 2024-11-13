#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-13 13:33:33 (ywatanabe)"
# File: ./mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/_BasicPlotMixin.py

from typing import List, Optional

import mngs
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from ....types import ArrayLike

"""
Functionality:
    * Provides basic plotting operations for matplotlib axes
Input:
    * Various data types for 2D visualization, line plots, density estimation, and confusion matrices
Output:
    * Modified matplotlib axis objects with plotted data
Prerequisites:
    * numpy, pandas, scipy.stats, matplotlib, mngs.plt
"""

from typing import Any, Dict, Optional, Tuple

from ....plt import ax as ax_module


class BasicPlotMixin:
    """Mixin class for basic plotting operations."""

    def imshow2d(self, arr_2d: ArrayLike, **kwargs) -> None:
        """Displays 2D array as an image.

        Parameters
        ----------
        arr_2d : ArrayLike
            2D array to display
        **kwargs : dict
            Additional arguments passed to mngs.plt.ax.imshow2d
        """
        method_name = "imshow2d"
        with self._no_tracking():
            self.axis = mngs.plt.ax.imshow2d(self.axis, arr_2d, **kwargs)
        out = pd.DataFrame(arr_2d)
        if kwargs.get("xyz", False):
            out = mngs.pd.to_xyz(out)
        self._track(
            kwargs.get("track"), kwargs.get("id"), method_name, out, None
        )

    def plot_(
        self,
        xx: Optional[ArrayLike] = None,
        yy: Optional[ArrayLike] = None,
        **kwargs,
    ) -> None:
        """Creates a line plot.

        Parameters
        ----------
        xx : ArrayLike, optional
            X-axis data points
        yy : ArrayLike, optional
            Y-axis data points
        **kwargs : dict
            Additional arguments passed to mngs.plt.ax.plot_
        """
        method_name = "plot_"
        with self._no_tracking():
            self.axis, df = mngs.plt.ax.plot_(
                self.axis, xx=xx, yy=yy, **kwargs
            )
        self._track(
            kwargs.get("track"), kwargs.get("id"), method_name, df, None
        )

    def kde(self, data: ArrayLike, **kwargs) -> None:
        """Plots kernel density estimation.

        Parameters
        ----------
        data : ArrayLike
            Input data for density estimation
        **kwargs : dict
            Additional plotting parameters including:
            - label: str, plot label
            - xlim: tuple, x-axis limits
            - cumulative: bool, whether to plot cumulative distribution
            - fill: bool, whether to fill under the curve
        """
        method_name = "kde"
        n_samples = (~np.isnan(data)).sum()
        if kwargs.get("label"):
            kwargs["label"] = f"{kwargs['label']} (n={n_samples})"
        xlim = kwargs.get("xlim", (data.min(), data.max()))
        xs = np.linspace(*xlim, int(1e3))
        density = gaussian_kde(data)(xs)
        density /= density.sum()
        if kwargs.get("cumulative"):
            density = np.cumsum(density)

        with self._no_tracking():
            if kwargs.get("fill"):
                self.axis.fill_between(xs, density, **kwargs)
            else:
                self.plot_(xx=xs, yy=density, label=kwargs.get("label"))

        out = pd.DataFrame(
            {"x": xs, "kde": density, "n": [len(data) for _ in range(len(xs))]}
        )
        self._track(
            kwargs.get("track"), kwargs.get("id"), method_name, out, None
        )

    def conf_mat(
        self,
        data: ArrayLike,
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        cbar: bool = True,
        cbar_kw: Dict[str, Any] = {},
        label_rotation_xy: Tuple[float, float] = (15, 15),
        x_extend_ratio: float = 1.0,
        y_extend_ratio: float = 1.0,
        bacc: bool = False,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Plots confusion matrix with optional balanced accuracy.

        Parameters
        ----------
        data : ArrayLike
            Confusion matrix data
        x_labels : List[str], optional
            Labels for x-axis
        y_labels : List[str], optional
            Labels for y-axis
        title : str
            Title of the confusion matrix
        cmap : str
            Colormap for the matrix
        cbar : bool
            Whether to show colorbar
        cbar_kw : dict
            Additional colorbar parameters
        label_rotation_xy : tuple
            Rotation angles for x and y labels
        x_extend_ratio : float
            Ratio to extend x-axis
        y_extend_ratio : float
            Ratio to extend y-axis
        bacc : bool
            Whether to calculate balanced accuracy
        """
        method_name = "conf_mat"
        with self._no_tracking():
            out = ax_module.conf_mat(
                self.axis,
                data,
                x_labels=x_labels,
                y_labels=y_labels,
                title=title,
                cmap=cmap,
                cbar=cbar,
                cbar_kw=cbar_kw,
                label_rotation_xy=label_rotation_xy,
                x_extend_ratio=x_extend_ratio,
                y_extend_ratio=y_extend_ratio,
                bacc=bacc,
                track=track,
                id=id,
                **kwargs,
            )
            bacc_val = None
            if bacc:
                self.axis, bacc_val = out
            else:
                self.axis = out
        out = data, bacc_val
        self._track(track, id, method_name, out, None)

    def rectangle(
        self,
        xx: float,
        yy: float,
        width: float,
        height: float,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Draws a rectangle on the plot.

        Parameters
        ----------
        xx : float
            X-coordinate of rectangle's lower-left corner
        yy : float
            Y-coordinate of rectangle's lower-left corner
        width : float
            Width of rectangle
        height : float
            Height of rectangle
        """
        method_name = "rectangle"
        with self._no_tracking():
            self.axis = ax_module.rectangle(
                self.axis, xx, yy, width, height, **kwargs
            )
        self._track(track, id, method_name, None, None)


# class BasicPlotMixin:
#     def imshow2d(self, arr_2d: np.ndarray, **kwargs) -> None:
#         method_name = "imshow2d"
#         with self._no_tracking():
#             self.axis = mngs.plt.ax.imshow2d(self.axis, arr_2d, **kwargs)
#         out = pd.DataFrame(arr_2d)
#         if kwargs.get("xyz", False):
#             out = mngs.pd.to_xyz(out)
#         self._track(
#             kwargs.get("track"), kwargs.get("id"), method_name, out, None
#         )

#     def plot_(
#         self,
#         xx: Optional[np.ndarray] = None,
#         yy: Optional[np.ndarray] = None,
#         **kwargs,
#     ) -> None:
#         method_name = "plot_"
#         with self._no_tracking():
#             self.axis, df = mngs.plt.ax.plot_(
#                 self.axis, xx=xx, yy=yy, **kwargs
#             )
#         self._track(
#             kwargs.get("track"), kwargs.get("id"), method_name, df, None
#         )

#     def kde(self, data: np.ndarray, **kwargs) -> None:
#         method_name = "kde"
#         n = (~np.isnan(data)).sum()
#         if kwargs.get("label"):
#             kwargs["label"] = f"{kwargs['label']} (n={n})"
#         xlim = kwargs.get("xlim", (data.min(), data.max()))
#         xs = np.linspace(*xlim, int(1e3))
#         density = gaussian_kde(data)(xs)
#         density /= density.sum()
#         if kwargs.get("cumulative"):
#             density = np.cumsum(density)
#         with self._no_tracking():
#             if kwargs.get("fill"):
#                 self.axis.fill_between(xs, density, **kwargs)
#             else:
#                 self.plot_(xx=xs, yy=density, label=kwargs.get("label"))
#         out = pd.DataFrame(
#             {"x": xs, "kde": density, "n": [len(data) for _ in range(len(xs))]}
#         )
#         self._track(
#             kwargs.get("track"), kwargs.get("id"), method_name, out, None
#         )

# EOF
