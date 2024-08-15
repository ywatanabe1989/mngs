#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-13 07:33:07 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/AxisWrapper.py

from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps

import mngs
import seaborn as sns
from mngs.gen import not_implemented
import pandas as pd
from ._to_sigma import to_sigma as _to_sigma
from scipy.stats import gaussian_kde
import numpy as np


class AxisWrapper:
    """
    A wrapper class for a Matplotlib axis that collects plotting data.
    """

    def __init__(self, fig, axis, track):
        """
        Initialize the AxisWrapper with a given axis and history reference.
        """
        self.fig = fig
        self.axis = axis
        self._ax_history = OrderedDict()
        self.track = track
        self.id = 0

    def get_figure(
        self,
    ):
        return self.fig

    def __getattr__(self, attr):
        if hasattr(self.axis, attr):
            original_attr = getattr(self.axis, attr)

            if callable(original_attr):

                @wraps(original_attr)
                def wrapper(*args, track=None, id=None, **kwargs):
                    results = original_attr(*args, **kwargs)
                    self._track(track, id, attr, args, kwargs)
                    return results

                return wrapper
            else:
                return original_attr
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    ################################################################################
    ## Tracking
    ################################################################################
    def _track(self, track, id, method_name, args, kwargs):
        if track is None:
            track = self.track
        if track:
            id = id if id is not None else self.id
            self.id += 1
            self._ax_history[id] = (id, method_name, args, kwargs)

    @contextmanager
    def _no_tracking(self):
        """Context manager to temporarily disable tracking."""
        original_track = self.track
        self.track = False
        try:
            yield
        finally:
            self.track = original_track

    @property
    def history(self):
        return {k: self._ax_history[k] for k in self._ax_history}

    def reset_history(self):
        self._ax_history = {}

    def to_sigma(self):
        """
        Export tracked plotting data to a DataFrame in SigmaPlot format.
        """
        df = _to_sigma(self.history)

        return df

    ################################################################################
    ## Plotting methods (other than matplotlib's original methods)
    ################################################################################
    def imshow2d(
        self,
        arr_2d,
        cbar=True,
        cbar_label=None,
        cbar_shrink=1.0,
        cbar_fraction=0.046,
        cbar_pad=0.04,
        cmap="viridis",
        aspect="auto",
        vmin=None,
        vmax=None,
        track=True,
        id=None,
        **kwargs,
    ):
        # Method name
        method_name = "imshow2d"

        # Plotting
        with self._no_tracking():
            self.axis = mngs.plt.ax.imshow2d(
                self.axis,
                arr_2d,
                cbar=cbar,
                cbar_label=cbar_label,
                cbar_shrink=cbar_shrink,
                cbar_fraction=cbar_fraction,
                cbar_pad=cbar_pad,
                cmap=cmap,
                aspect=aspect,
                vmin=vmin,
                vmax=vmax,
            )

        # Tracking
        out = pd.DataFrame(arr_2d)
        self._track(track, id, method_name, out, None)

    def rectangle(self, xx, yy, ww, hh, track=True, id=None, **kwargs):
        # Method name
        method_name = "rectangle"

        # Plotting
        with self._no_tracking():
            self.axis = mngs.plt.ax.rectangle(
                self.axis, xx, yy, ww, hh, **kwargs
            )

        # Tracking
        out = None
        self._track(track, id, method_name, out, None)

    def plot_with_ci(
        self,
        xx,
        mean,
        std,
        track=True,
        id=None,
        **kwargs,
    ):
        # Method name
        method_name = "plot_with_ci"

        # Plotting
        with self._no_tracking():
            self.axis = mngs.plt.ax.plot_with_ci(
                self.axis, xx, mean, std, **kwargs
            )

        # Tracking
        out = (xx, mean, std)
        self._track(track, id, method_name, out, None)

    def fillv(
        self,
        starts,
        ends,
        color="red",
        alpha=0.2,
        track=True,
        id=None,
        **kwargs,
    ):
        # Method name
        method_name = "fillv"

        self.axis = mngs.plt.ax.fillv(
            self.axis, starts, ends, color=color, alpha=alpha
        )

        # Tracking
        out = (starts, ends)
        self._track(track, id, method_name, out, None)

    def raster(self, positions, time=None, track=True, id=None, **kwargs):
        # Method name
        method_name = "raster"

        # Plotting
        with self._no_tracking():
            self.axis, df = mngs.plt.ax.raster(
                self.axis, positions, time=time, **kwargs
            )
        out = df

        # Tracking
        self._track(track, id, method_name, out, None)

    def kde(self, data, track=True, id=None, xlim=None, **kwargs):
        # Method name
        method_name = "kde"

        xlim = xlim if xlim is not None else (data.min(), data.max())
        xs = np.linspace(*xlim, int(1e3))
        density = gaussian_kde(data)(xs)
        density /= density.sum()

        # Plotting
        with self._no_tracking():
            if kwargs.get("fill"):
                self.axis.fill_between(xs, density, **kwargs)
            else:
                self.axis.plot(xs, density, **kwargs)

        # Tracking
        out = pd.DataFrame(
            {
                "x": xs,
                "kde": density,
            }
        )
        self._track(track, id, method_name, out, None)

    def ecdf(self, data, track=True, id=None, **kwargs):
        # Method name
        method_name = "ecdf"

        # Plotting
        with self._no_tracking():
            self.axis, df = mngs.plt.ax.ecdf(self.axis, data, **kwargs)
        out = df

        # Tracking
        self._track(track, id, method_name, out, None)

    @not_implemented
    def joyplot(
        self,
        track=True,
        id=None,
    ):
        # Method name
        method_name = "joyplot"

        # Plotting
        __import__("ipdb").set_trace()

        # Tracking
        out = None
        self._track(track, id, method_name, out, None)

    ################################################################################
    ## Seaborn-wrappers
    ################################################################################
    def _sns_base(self, method_name, *args, track=True, id=None, **kwargs):
        sns_method_name = method_name.split("sns_")[-1]

        with self._no_tracking():
            plot_func = getattr(sns, sns_method_name)
            self.axis = plot_func(ax=self.axis, *args, **kwargs)

        # Track the plot if required
        self._track(track, id, method_name, args, kwargs)

    def sns_barplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_barplot", *args, track=track, id=id, **kwargs)

    def sns_boxplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_boxplot", *args, track=track, id=id, **kwargs)

    def sns_heatmap(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_heatmap", *args, track=track, id=id, **kwargs)

    def sns_histplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_histplot", *args, track=track, id=id, **kwargs)

    def sns_kdeplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_kdeplot", *args, track=track, id=id, **kwargs)

    def sns_lineplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_lineplot", *args, track=track, id=id, **kwargs)

    def sns_pairplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_pairplot", *args, track=track, id=id, **kwargs)

    def sns_scatterplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_scatterplot", *args, track=track, id=id, **kwargs)

    def sns_violinplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_violinplot", *args, track=track, id=id, **kwargs)

    def sns_jointplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_jointplot", *args, track=track, id=id, **kwargs)

    ################################################################################
    ## Adjusting methods
    ################################################################################
    def set_xyt(
        self,
        xlabel=None,
        ylabel=None,
        title=None,
    ):
        self.axis = mngs.plt.ax.set_xyt(
            self.axis,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

    def set_supxyt(
        self,
        xlabel=None,
        ylabel=None,
        title=None,
    ):
        self.axis = mngs.plt.ax.set_supxyt(
            self.axis,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

    def set_ticks(
        self,
        xvals=None,
        xticks=None,
        yvals=None,
        yticks=None,
        **kwargs,
    ):

        self.axis = mngs.plt.ax.set_ticks(
            self.axis,
            xvals=xvals,
            xticks=xticks,
            yvals=yvals,
            yticks=yticks,
        )

    def set_n_ticks(self, n_xticks=4, n_yticks=4):
        self.axis = mngs.plt.ax.set_n_ticks(
            self.axis, n_xticks=n_xticks, n_yticks=n_yticks
        )

    def hide_spines(
        self,
        top=True,
        bottom=True,
        left=True,
        right=True,
        ticks=True,
        labels=True,
    ):
        self.axis = mngs.plt.ax.hide_spines(
            self.axis,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            ticks=ticks,
            labels=labels,
        )

    def extend(self, x_ratio=1.0, y_ratio=1.0):
        self.axis = mngs.plt.ax.extend(
            self.axis, x_ratio=x_ratio, y_ratio=y_ratio
        )

    def shift(self, dx=0, dy=0):
        self.axis = mngs.plt.ax.shift(self.axis, dx=dx, dy=dy)