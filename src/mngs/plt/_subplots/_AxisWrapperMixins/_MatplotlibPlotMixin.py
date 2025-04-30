#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 08:57:40 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/_MatplotlibPlotMixin.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_AxisWrapperMixins/_MatplotlibPlotMixin.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/_BasicPlotMixin.py"

from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from ....pd import to_xyz
from ....plt import ax as ax_module
from ....types import ArrayLike


class MatplotlibPlotMixin:
    """Mixin class for basic plotting operations."""

    @wraps(ax_module.plot_image)
    def plot_image(self, arr_2d: ArrayLike, **kwargs) -> None:
        method_name = "plot_image"
        with self._no_tracking():
            self._axis_mpl = ax_module.plot_image(
                self._axis_mpl, arr_2d, **kwargs
            )
        out = pd.DataFrame(arr_2d)
        # When xyz format specified
        if kwargs.get("xyz", False):
            out = to_xyz(out)
        self._track(
            kwargs.get("track"), kwargs.get("id"), method_name, out, None
        )

    def kde(self, data: ArrayLike, **kwargs) -> None:
        method_name = "kde"
        n_samples = (~np.isnan(data)).sum()
        if kwargs.get("label"):
            kwargs["label"] = f"{kwargs['label']} (n={n_samples})"
        xlim = kwargs.get("xlim", (np.nanmin(data), np.nanmax(data)))
        xx = np.linspace(*xlim, int(1e3))
        density = gaussian_kde(data)(xx)
        density /= density.sum()
        if kwargs.get("cumulative"):
            density = np.cumsum(density)
        with self._no_tracking():
            if kwargs.get("fill"):
                self._axis_mpl.fill_between(xx, density, **kwargs)
            else:
                self._axis_mpl.plot(xx, density, label=kwargs.get("label"))
        out = pd.DataFrame(
            {
                "x": xx,
                "kde": density,
                "n": [len(data) for ii in range(len(xx))],
            }
        )
        self._track(
            kwargs.get("track"), kwargs.get("id"), method_name, out, None
        )

    def plot_conf_mat(
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
        method_name = "plot_conf_mat"
        with self._no_tracking():
            out = ax_module.plot_conf_mat(
                self._axis_mpl,
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
            self._axis_mpl, bacc_val = out
        else:
            self._axis_mpl = out
        out = data, bacc_val
        self._track(track, id, method_name, out, None)

    @wraps(ax_module.plot_rectangle)
    def plot_rectangle(
        self,
        xx: float,
        yy: float,
        width: float,
        height: float,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        method_name = "plot_rectangle"
        with self._no_tracking():
            self._axis_mpl = ax_module.plot_rectangle(
                self._axis_mpl, xx, yy, width, height, **kwargs
            )
        self._track(track, id, method_name, None, None)

    @wraps(ax_module.plot_fillv)
    def plot_fillv(
        self,
        starts: ArrayLike,
        ends: ArrayLike,
        color: str = "red",
        alpha: float = 0.2,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        method_name = "plot_fillv"
        self._axis_mpl = ax_module.plot_fillv(
            self._axis_mpl, starts, ends, color=color, alpha=alpha
        )
        out = (starts, ends)
        self._track(track, id, method_name, out, None)

    def boxplot_(
        self,
        data: ArrayLike,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        method_name = "boxplot_"
        _data = data.copy()
        n = len(data)
        if kwargs.get("label"):
            kwargs["label"] = kwargs["label"] + f" (n={n})"
        with self._no_tracking():
            self._axis_mpl.boxplot(data, **kwargs)
        out = pd.DataFrame(
            {
                "data": _data,
                "n": [n for ii in range(len(data))],
            }
        )
        self._track(track, id, method_name, out, None)

    @wraps(ax_module.plot_raster)
    def raster(
        self,
        positions: List[ArrayLike],
        time: Optional[ArrayLike] = None,
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        method_name = "raster"
        if colors is None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        if len(colors) < len(positions):
            colors = colors * (len(positions) // len(colors) + 1)
        with self._no_tracking():
            for i_, (pos, color) in enumerate(zip(positions, colors)):
                label = (
                    labels[i_]
                    if labels is not None and i_ < len(labels)
                    else None
                )
                self._axis_mpl.eventplot(
                    pos, colors=color, label=label, **kwargs
                )
            if labels is not None:
                self._axis_mpl.legend()
        df = ax_module.plot_raster(self._axis_mpl, positions, time=time)[1]
        if id is not None:
            df.columns = [f"{id}_{method_name}_{col}" for col in df.columns]
        out = df
        self._track(track, id, method_name, out, None)

    @wraps(ax_module.plot_ecdf)
    def plot_ecdf(
        self,
        data: ArrayLike,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        method_name = "plot_ecdf"
        with self._no_tracking():
            self._axis_mpl, df = ax_module.plot_ecdf(
                self._axis_mpl, data, **kwargs
            )
        out = df
        self._track(track, id, method_name, out, None)

    @wraps(ax_module.plot_vertical_joyplot)
    def plot_vertical_joyplot(
        self,
        data: ArrayLike,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        method_name = "plot_vertical_joyplot"
        with self._no_tracking():
            self._axis_mpl = ax_module.plot_vertical_joyplot(
                self._axis_mpl, data, **kwargs
            )
        out = data
        self._track(track, id, method_name, out, None)

    @wraps(ax_module.plot_horizontal_joyplot)
    def plot_horizontal_joyplot(
        self,
        data: ArrayLike,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        method_name = "plot_horizontal_joyplot"
        with self._no_tracking():
            self._axis_mpl = ax_module.plot_horizontal_joyplot(
                self._axis_mpl, data, **kwargs
            )
        out = data
        self._track(track, id, method_name, out, None)

# EOF