#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-07 20:46:07 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/AxisWrapper.py

from functools import wraps

import mngs
import pandas as pd
import seaborn as sns

from ._format_sns_args_for_sigmaplot import format_sns_args_for_sigmaplot
from ._to_sigmaplot_format import to_sigmaplot_format


class AxisWrapper:
    """
    A wrapper class for a Matplotlib axis that collects plotting data.
    """

    def __init__(self, axis, history, track):
        """
        Initialize the AxisWrapper with a given axis and history reference.
        """
        self.axis = axis
        self._history = history
        self.track = track
        self.id = 0

    def __getattr__(self, attr):
        """
        Wrap the axis attribute access to collect plot calls or return the attribute directly.
        """
        original_attr = getattr(self.axis, attr)

        if callable(original_attr):

            @wraps(original_attr)
            def wrapper(
                *args, id=None, track=True, n_xticks=4, n_yticks=4, **kwargs
            ):
                result = original_attr(*args, **kwargs)

                if attr in ["plot", "scatter", "plot_with_ci"]:
                    self.axis = mngs.plt.ax.set_n_ticks(
                        self.axis, n_xticks=n_xticks, n_yticks=n_yticks
                    )

                self._track(track, id, attr, args, kwargs)

                return result

            return wrapper

        else:
            return original_attr

    ################################################################################
    ## Basic methods
    ################################################################################
    def _track(self, track, id, method_name, args, kwargs):
        if self.track:
            if id is not None:
                id = self.id
                self.id += 1
            self._history[id] = (id, method_name, args, kwargs)

    @property
    def history(self):
        return {k: self._history[k] for k in self._history}

    def reset_history(self):
        self._history = {}

    def to_sigma(self):
        """
        Convert the axis history to a sigma format DataFrame.
        """
        try:
            df = pd.concat(
                [to_sigmaplot_format(v) for v in self.history.values()], axis=1
            )

        except Exception as e:
            print(e)
            df = pd.DataFrame()

        return df

    ################################################################################
    ## Plotting methods (other than matplotlib's original methods)
    ################################################################################
    # def joyplot():
    #     pass

    def plot_with_ci(
        self, xx, mean, std, label=None, alpha=0.5, track=True, id=None
    ):
        self.axis = mngs.plt.ax.plot_with_ci(
            self.axis, xx, mean, std, label=label, alpha=alpha
        )
        self._track(track, id, "fill_between", (xx, mean, std), None)

    def raster(self, positions, time=None, track=True, id=None, **kwargs):
        self.axis, df = mngs.plt.ax.raster(
            self.axis, positions, time=time, **kwargs
        )
        self._track(track, id, "raster", df, None)

    def rectangle(self, xx, yy, ww, hh, **kwargs):
        self.axis = mngs.plt.ax.rectangle(self.axis, xx, yy, ww, hh, **kwargs)

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

        self._track(track, id, "imshow2d", arr_2d, None)

    ################################################################################
    ## Seaborn-wrappers
    ################################################################################
    def _sns_base(self, method_name, *args, track=True, id=None, **kwargs):
        actual_method_name = method_name.split("sns_")[-1]
        try:
            plot_func = getattr(sns, actual_method_name)
            self.axis = plot_func(*args, **kwargs)
        except AttributeError:
            raise ValueError(
                f"{actual_method_name} is not a valid seaborn function"
            )

        # Format arguments for tracking
        formatted_df = format_sns_args_for_sigmaplot(
            actual_method_name, *args, **kwargs
        )

        # Track the plot if required
        self._track(track, id, method_name, formatted_df, None)

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
