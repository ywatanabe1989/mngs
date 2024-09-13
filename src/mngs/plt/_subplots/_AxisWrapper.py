#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-12 12:11:19 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/AxisWrapper.py

from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps

import mngs
import numpy as np
import pandas as pd
import seaborn as sns
from mngs.gen import not_implemented
from scipy.stats import gaussian_kde

from ._to_sigma import to_sigma as _to_sigma


def sns_copy_doc(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    wrapper.__doc__ = getattr(sns, func.__name__.split("sns_")[-1]).__doc__
    return wrapper


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

    @property
    def flat(self):
        return [self.axis]

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
        n=None,
        track=True,
        id=None,
        **kwargs,
    ):
        # Method name
        method_name = "plot_with_ci"

        # Plotting
        with self._no_tracking():
            self.axis = mngs.plt.ax.plot_with_ci(
                self.axis, xx, mean, std, n=n, **kwargs
            )

        # Tracking
        out = (xx, mean, std, n)
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
            self.axis, df = mngs.plt.ax.raster_plot(
                self.axis, positions, time=time, **kwargs
            )
        if id is not None:
            df.columns = [f"{id}_{method_name}_{col}" for col in df.columns]
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

    def joyplot(
        self,
        data,
        track=True,
        id=None,
        **kwargs,
    ):
        # Method name
        method_name = "joyplot"

        # Plotting
        with self._no_tracking():
            self.axis = mngs.plt.ax.joyplot(self.axis, data, **kwargs)

        # Tracking
        out = data
        self._track(track, id, method_name, out, None)

    ################################################################################
    ## Seaborn-wrappers
    ################################################################################
    def _sns_base(
        self, method_name, *args, track=True, track_obj=None, id=None, **kwargs
    ):
        sns_method_name = method_name.split("sns_")[-1]

        with self._no_tracking():
            sns_plot_fn = getattr(sns, sns_method_name)

            if kwargs.get("hue_colors"):
                kwargs = mngs.gen.alternate_kwarg(
                    kwargs, primary_key="palette", alternate_key="hue_colors"
                )

            self.axis = sns_plot_fn(ax=self.axis, *args, **kwargs)

        # Track the plot if required
        track_obj = track_obj if track_obj is not None else args
        self._track(track, id, method_name, track_obj, kwargs)

    def _sns_base_xyhue(
        self, method_name, *args, track=True, id=None, **kwargs
    ):
        df = kwargs.get("data")
        x, y, hue = kwargs.get("x"), kwargs.get("y"), kwargs.get("hue")

        track_obj = (
            self._sns_prepare_xyhue(df, x, y, hue) if df is not None else None
        )

        self._sns_base(
            method_name,
            *args,
            track=track,
            track_obj=track_obj,
            id=id,
            **kwargs,
        )

    # def _sns_prepare_xyhue(self, df, x, y, hue=None):
    #     if x and y:
    #         if hue:
    #             pivoted_data = df.pivot_table(
    #                 values=y, index=df.index, columns=[x, hue], aggfunc="first"
    #             )
    #             pivoted_data.columns = [
    #                 f"{col[0]}-{col[1]}" for col in pivoted_data.columns
    #             ]
    #         else:
    #             pivoted_data = df.pivot_table(
    #                 values=y, index=df.index, columns=x, aggfunc="first"
    #             )
    #         return pivoted_data
    #     return None

    # def _sns_prepare_xyhue(self, df, x, y, hue=None):
    #     if x is None and y is None:
    #         return df
    #     elif x is None:
    #         return df[[y]]
    #     elif y is None:
    #         return df[[x]]
    #     else:
    #         if hue:
    #             pivoted_data = df.pivot_table(
    #                 values=y, index=df.index, columns=[x, hue], aggfunc="first"
    #             )
    #             pivoted_data.columns = [
    #                 f"{col[0]}-{col[1]}" for col in pivoted_data.columns
    #             ]
    #         else:
    #             pivoted_data = df.pivot_table(
    #                 values=y, index=df.index, columns=x, aggfunc="first"
    #             )
    #         return pivoted_data

    # @sns_copy_doc
    # def sns_barplot(self, *args, track=True, id=None, **kwargs):
    #     self._sns_base_xyhue(
    #         "sns_barplot", *args, track=track, id=id, **kwargs
    #     )

    # @sns_copy_doc
    # def sns_boxplot(self, *args, track=True, id=None, **kwargs):
    #     self._sns_base_xyhue(
    #         "sns_boxplot", *args, track=track, id=id, **kwargs
    #     )

    def _sns_prepare_xyhue(
        self, data=None, x=None, y=None, hue=None, **kwargs
    ):
        if x is None and y is None:
            return data
        elif x is None:
            return data[[y]]
        elif y is None:
            return data[[x]]
        else:
            if hue:
                pivoted_data = data.pivot_table(
                    values=y,
                    index=data.index,
                    columns=[x, hue],
                    aggfunc="first",
                )
                pivoted_data.columns = [
                    f"{col[0]}-{col[1]}" for col in pivoted_data.columns
                ]
            else:
                pivoted_data = data.pivot_table(
                    values=y, index=data.index, columns=x, aggfunc="first"
                )
            return pivoted_data

    @sns_copy_doc
    def sns_barplot(
        self, data=None, x=None, y=None, track=True, id=None, **kwargs
    ):
        self._sns_base_xyhue(
            "sns_barplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_boxplot(
        self, data=None, x=None, y=None, track=True, id=None, **kwargs
    ):
        self._sns_base_xyhue(
            "sns_boxplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_heatmap(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_heatmap", *args, track=track, id=id, **kwargs)

    @sns_copy_doc
    def sns_histplot(
        self, data=None, x=None, y=None, track=True, id=None, **kwargs
    ):
        self._sns_base_xyhue(
            "sns_histplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_kdeplot(
        self, data=None, x=None, y=None, track=True, id=None, **kwargs
    ):
        self._sns_base_xyhue(
            "sns_kdeplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_lineplot(
        self, data=None, x=None, y=None, track=True, id=None, **kwargs
    ):
        self._sns_base_xyhue(
            "sns_lineplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_pairplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_pairplot", *args, track=track, id=id, **kwargs)

    @sns_copy_doc
    def sns_scatterplot(
        self, data=None, x=None, y=None, track=True, id=None, **kwargs
    ):
        self._sns_base_xyhue(
            "sns_scatterplot",
            data=data,
            x=x,
            y=y,
            track=track,
            id=id,
            **kwargs,
        )

    @sns_copy_doc
    def sns_violinplot(
        self, data=None, x=None, y=None, track=True, id=None, **kwargs
    ):
        self._sns_base_xyhue(
            "sns_violinplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_jointplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_jointplot", *args, track=track, id=id, **kwargs)

    ################################################################################
    ## Adjusting methods
    ################################################################################
    def set_xyt(
        self,
        x=None,
        y=None,
        t=None,
        format_labels=True,
    ):
        self.axis = mngs.plt.ax.set_xyt(
            self.axis,
            x=x,
            y=y,
            t=t,
            format_labels=format_labels,
        )

    def set_supxyt(
        self,
        xlabel=None,
        ylabel=None,
        title=None,
        format_labels=True,
    ):
        self.axis = mngs.plt.ax.set_supxyt(
            self.axis,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            format_labels=format_labels,
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
