#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-17 12:00:05 (ywatanabe)"
# File: ./mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/_BasicPlotMixin.py

THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/_BasicPlotMixin.py"

from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
# from numpy.typing import ArrayLike
from scipy.stats import gaussian_kde

from ....pd import to_xyz
from ....plt import ax as ax_module
from ....types import ArrayLike

class BasicPlotMixin:
    """Mixin class for basic plotting operations."""

    @wraps(ax_module.imshow2d)
    def imshow2d(self, arr_2d: ArrayLike, **kwargs) -> None:
        method_name = "imshow2d"
        with self._no_tracking():
            self.axis = ax_module.imshow2d(self.axis, arr_2d, **kwargs)
        out = pd.DataFrame(arr_2d)
        if kwargs.get("xyz", False):
            out = to_xyz(out)
        self._track(kwargs.get("track"), kwargs.get("id"), method_name, out, None)


    @wraps(ax_module.plot_)
    def plot_(
        self,
        data: ArrayLike,
        xx: Optional[ArrayLike] = None,
        yy: Optional[ArrayLike] = None,
        line: Optional[str] = None,
        fill: Optional[str] = None,
        n: Optional[Union[int, float, ArrayLike]] = None,
        alpha: float = 0.3,
        **kwargs,
    ) -> None:
        method_name = "plot_"
        with self._no_tracking():
            self.axis, df = ax_module.plot_(
                self.axis,
                data=data,
                xx=xx,
                yy=yy,
                line=line,
                fill=fill,
                n=n,
                alpha=alpha,
                **kwargs,
            )
        self._track(kwargs.get("track"), kwargs.get("id"), method_name, df, None)


    # @wraps(ax_module.kde)
    def kde(self, data: ArrayLike, **kwargs) -> None:
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
                # self.axis.plot_(xs, density, **kwargs)
            else:
                self.plot_(xx=xs, yy=density, label=kwargs.get("label"))

        out = pd.DataFrame(
            {"x": xs, "kde": density, "n": [len(data) for _ in range(len(xs))]}
        )
        self._track(kwargs.get("track"), kwargs.get("id"), method_name, out, None)

    # @wraps(ax_module.conf_mat)
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

    @wraps(ax_module.rectangle)
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
        method_name = "rectangle"
        with self._no_tracking():
            self.axis = ax_module.rectangle(
                self.axis, xx, yy, width, height, **kwargs
            )
        self._track(track, id, method_name, None, None)

    @wraps(ax_module.fillv)
    def fillv(
        self,
        starts: ArrayLike,
        ends: ArrayLike,
        color: str = "red",
        alpha: float = 0.2,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:

        method_name = "fillv"
        self.axis = ax_module.fillv(
            self.axis, starts, ends, color=color, alpha=alpha
        )
        out = (starts, ends)
        self._track(track, id, method_name, out, None)

    # @wraps(ax_module.boxplot_)
    def boxplot_(
        self,
        data: ArrayLike,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs
    ) -> None:
        method_name = "boxplot_"
        _data = data.copy()
        n = len(data)

        if kwargs.get("label"):
            kwargs["label"] = kwargs["label"] + f" (n={n})"

        with self._no_tracking():
            self.axis.boxplot(data, **kwargs)

        out = pd.DataFrame(
            {
                "data": _data,
                "n": [n for _ in range(len(data))],
            }
        )
        self._track(track, id, method_name, out, None)

    # @wraps(ax_module.raster)
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
            for i, (pos, color) in enumerate(zip(positions, colors)):
                label = labels[i] if labels is not None and i < len(labels) else None
                self.axis.eventplot(pos, colors=color, label=label, **kwargs)

            if labels is not None:
                self.axis.legend()

            df = ax_module.raster_plot(self.axis, positions, time=time)[1]

        if id is not None:
            df.columns = [f"{id}_{method_name}_{col}" for col in df.columns]
        out = df
        self._track(track, id, method_name, out, None)

    @wraps(ax_module.ecdf)
    def ecdf(
        self,
        data: ArrayLike,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs
    ) -> None:
        method_name = "ecdf"
        with self._no_tracking():
            self.axis, df = ax_module.ecdf(self.axis, data, **kwargs)
        out = df
        self._track(track, id, method_name, out, None)

    @wraps(ax_module.joyplot)
    def joyplot(
        self,
        data: ArrayLike,
        track: bool = True,
        id: Optional[str] = None,
        **kwargs,
    ) -> None:
        method_name = "joyplot"
        with self._no_tracking():
            self.axis = ax_module.joyplot(self.axis, data, **kwargs)
        out = data
        self._track(track, id, method_name, out, None)

# EOF
