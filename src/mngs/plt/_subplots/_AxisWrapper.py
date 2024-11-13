#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-13 13:31:52 (ywatanabe)"
# File: ./mngs_repo/src/mngs/plt/_subplots/_AxisWrapper.py

from collections import OrderedDict
from functools import wraps

import matplotlib.pyplot as plt
import pandas as pd

from ...plt import ax as ax_module


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

    # ################################################################################
    # ## Plotting methods (other than matplotlib's original methods)
    # ################################################################################
    # def imshow2d(
    #     self,
    #     arr_2d,
    #     cbar=True,
    #     cbar_label=None,
    #     cbar_shrink=1.0,
    #     cbar_fraction=0.046,
    #     cbar_pad=0.04,
    #     cmap="viridis",
    #     aspect="auto",
    #     vmin=None,
    #     vmax=None,
    #     track=True,
    #     id=None,
    #     xyz=False,
    #     **kwargs,
    # ):
    #     # Method name
    #     method_name = "imshow2d"

    #     # Plotting
    #     with self._no_tracking():
    #         self.axis = ax_module.imshow2d(
    #             self.axis,
    #             arr_2d,
    #             cbar=cbar,
    #             cbar_label=cbar_label,
    #             cbar_shrink=cbar_shrink,
    #             cbar_fraction=cbar_fraction,
    #             cbar_pad=cbar_pad,
    #             cmap=cmap,
    #             aspect=aspect,
    #             vmin=vmin,
    #             vmax=vmax,
    #         )

    #     # Tracking
    #     out = pd.DataFrame(arr_2d)
    #     if xyz:
    #         out = mngs.pd.to_xyz(out)

    #     self._track(track, id, method_name, out, None)

    # def conf_mat(
    #     self,
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
    #     track=True,
    #     id=None,
    #     **kwargs,
    # ):
    #     # Method name
    #     method_name = "conf_mat"

    #     # Plotting
    #     with self._no_tracking():
    #         out = ax_module.conf_mat(
    #             self.axis,
    #             data,
    #             x_labels=x_labels,
    #             y_labels=y_labels,
    #             title=title,
    #             cmap=cmap,
    #             cbar=cbar,
    #             cbar_kw=cbar_kw,
    #             label_rotation_xy=label_rotation_xy,
    #             x_extend_ratio=x_extend_ratio,
    #             y_extend_ratio=y_extend_ratio,
    #             bacc=bacc,
    #             track=track,
    #             id=id,
    #             **kwargs,
    #         )
    #         bacc_val = None

    #         if bacc:
    #             self.axis, bacc_val = out
    #         else:
    #             self.axis = out

    #     # Tracking
    #     out = data, bacc_val
    #     self._track(track, id, method_name, out, None)

    # def rectangle(self, xx, yy, ww, hh, track=True, id=None, **kwargs):
    #     # Method name
    #     method_name = "rectangle"

    #     # Plotting
    #     with self._no_tracking():
    #         self.axis = ax_module.rectangle(
    #             self.axis, xx, yy, ww, hh, **kwargs
    #         )

    #     # Tracking
    #     out = None
    #     self._track(track, id, method_name, out, None)

    # def plot_(
    #     self,
    #     xx=None,
    #     yy=None,
    #     mean=None,
    #     median=None,
    #     std=None,
    #     ci=None,
    #     iqr=None,
    #     n=None,
    #     alpha=0.3,
    #     track=True,
    #     id=None,
    #     **kwargs,
    # ):
    #     # Method
    #     method_name = "plot_"

    #     # Plotting
    #     with self._no_tracking():
    #         self.axis, df = ax_module.plot_(
    #             self.axis,
    #             xx=xx,
    #             yy=yy,
    #             mean=mean,
    #             median=median,
    #             std=std,
    #             ci=ci,
    #             iqr=iqr,
    #             n=n,
    #             alpha=alpha,
    #             **kwargs,
    #         )

    #     # Tracking
    #     out = df
    #     self._track(track, id, method_name, out, None)

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

        self.axis = ax_module.fillv(
            self.axis, starts, ends, color=color, alpha=alpha
        )

        # Tracking
        out = (starts, ends)
        self._track(track, id, method_name, out, None)

    def boxplot_(self, data, track=True, id=None, **kwargs):
        # Method name
        method_name = "boxplot_"

        # Deep Copy
        _data = data.copy()

        # # NaN Handling
        # data = np.hstack(data)
        # data = data[~np.isnan(data)]

        n = len(data)

        if kwargs.get("label"):
            kwargs["label"] = kwargs["label"] + f" (n={n})"

        # Plotting
        with self._no_tracking():
            self.axis.boxplot(data, **kwargs)

        out = pd.DataFrame(
            {
                "data": _data,
                "n": [n for _ in range(len(data))],
            }
        )

        # Tracking
        self._track(track, id, method_name, out, None)

    # def raster(self, positions, time=None, track=True, id=None, **kwargs):
    #     # Method name
    #     method_name = "raster"

    #     # Plotting
    #     with self._no_tracking():
    #         self.axis, df = ax_module.raster_plot(
    #             self.axis, positions, time=time, **kwargs
    #         )
    #     if id is not None:
    #         df.columns = [f"{id}_{method_name}_{col}" for col in df.columns]
    #     out = df

    #     # Tracking
    #     self._track(track, id, method_name, out, None)
    def raster(
        self,
        positions,
        time=None,
        labels=None,
        colors=None,
        track=True,
        id=None,
        **kwargs,
    ):
        """
        Create raster plot with optional labels and colors.

        Parameters
        ----------
        positions : list of lists
            Event positions for each channel
        time : array-like, optional
            Time indices
        labels : list, optional
            Labels for each channel
        colors : list, optional
            Colors for each channel
        track : bool, default True
            Whether to track the plotting data
        id : str, optional
            Identifier for tracking
        **kwargs : dict
            Additional arguments for eventplot
        """
        # Method name
        method_name = "raster"

        # Handle colors
        if colors is None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        if len(colors) < len(positions):
            colors = colors * (len(positions) // len(colors) + 1)

        # Plotting
        with self._no_tracking():
            for i, (pos, color) in enumerate(zip(positions, colors)):
                label = (
                    labels[i]
                    if labels is not None and i < len(labels)
                    else None
                )
                self.axis.eventplot(pos, colors=color, label=label, **kwargs)

            if labels is not None:
                self.axis.legend()

            df = ax_module.raster_plot(self.axis, positions, time=time)[1]

        if id is not None:
            df.columns = [f"{id}_{method_name}_{col}" for col in df.columns]
        out = df

        # Tracking
        self._track(track, id, method_name, out, None)

    # def kde(self, data, track=True, id=None, xlim=None, **kwargs):
    #     # Method name
    #     method_name = "kde"

    #     # NaN Handling
    #     n = (~np.isnan(data)).sum()

    #     if kwargs.get("label"):
    #         kwargs["label"] = kwargs["label"] + f" (n={n})"

    #     # x axis
    #     xlim = xlim if xlim is not None else (data.min(), data.max())
    #     xs = np.linspace(*xlim, int(1e3))

    #     # KDE
    #     density = gaussian_kde(data)(xs)
    #     density /= density.sum()

    #     # Cumulative
    #     if kwargs.get("cumulative"):
    #         density = np.cumsum(density)

    #     # Plotting
    #     with self._no_tracking():
    #         if kwargs.get("fill"):
    #             self.axis.fill_between(xs, density, **kwargs)
    #         else:
    #             self.plot_(xx=xs, yy=density, label=kwargs.get("label"))

    #     # Tracking
    #     out = pd.DataFrame(
    #         {"x": xs, "kde": density, "n": [len(data) for _ in range(len(xs))]}
    #     )
    #     self._track(track, id, method_name, out, None)

    def ecdf(self, data, track=True, id=None, **kwargs):
        # Method name
        method_name = "ecdf"

        # Plotting
        with self._no_tracking():
            self.axis, df = ax_module.ecdf(self.axis, data, **kwargs)
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
            self.axis = ax_module.joyplot(self.axis, data, **kwargs)

        # Tracking
        out = data
        self._track(track, id, method_name, out, None)

    # @sns_copy_doc
    # def sns_violinplot(
    #     self, data=None, x=None, y=None, track=True, id=None, half=False, **kwargs
    # ):
    #     if half:
    #         return self.sns_categorical_kde_plot(data=data, x=x, y=y, track=track, id=id, **kwargs)
    #     else:
    #         self._sns_base_xyhue(
    #             "sns_violinplot", data=data, x=x, y=y, track=track, id=id, **kwargs
    #         )

    # @sns_copy_doc
    # def sns_violinplot(
    #     self, data=None, x=None, y=None, track=True, id=None, half=False, **kwargs
    # ):
    #     if half:
    #         # Add a fake hue column
    #         data = data.copy()
    #         original_hue = kwargs.get('hue')
    #         data['_fake_hue'] = data[original_hue] + '_right'
    #         kwargs['hue'] = '_fake_hue'
    #         kwargs['split'] = True

    #         if 'hue_order' in kwargs:
    #             kwargs['hue_order'] = [h + '_right' for h in kwargs['hue_order']]

    #         if 'hue_colors' in kwargs:
    #             kwargs['hue_colors'] = kwargs['hue_colors'] * 2

    #     self._sns_base_xyhue(
    #         "sns_violinplot", data=data, x=x, y=y, track=track, id=id, **kwargs
    #     )

    # def sns_categorical_kde_plot(self, data=None, x=None, y=None, track=True, id=None, **kwargs):
    #     self._sns_base_xyhue(
    #         "sns_kdeplot", data=data, x=x, y=y, multiple="stack", track=track, id=id, **kwargs
    #     )

    # @sns_copy_doc
    # def sns_jointplot(self, *args, track=True, id=None, **kwargs):
    #     self._sns_base("sns_jointplot", *args, track=track, id=id, **kwargs)

    # @sns_copy_doc
    # def sns_half_violin(self, data=None, x=None, y=None, hue=None, track=True, id=None, **kwargs):
    #     # Method name
    #     method_name = "sns_half_violin"

    #     # Plotting
    #     with self._no_tracking():
    #         self.axis = mngs.plt.half_violin(self.axis, data=data, x=x, y=y, hue=hue, **kwargs)

    #     # Tracking
    #     track_obj = self._sns_prepare_xyhue(data, x, y, hue)
    #     self._track(track, id, method_name, track_obj, kwargs)

    #     return self.axis

    # ################################################################################
    # ## Adjusting methods
    # ################################################################################
    # def rotate_labels(self, x=30, y=30, x_ha="right", y_ha="center"):
    #     self.axis = ax_module.rotate_labels(
    #         self.axis, x=x, y=y, x_ha=x_ha, y_ha=y_ha
    #     )

    # def legend(self, loc="upper left"):
    #     return self.axis.legend(loc=loc)

    # def set_xyt(
    #     self,
    #     x=None,
    #     y=None,
    #     t=None,
    #     format_labels=True,
    # ):
    #     self.axis = ax_module.set_xyt(
    #         self.axis,
    #         x=x,
    #         y=y,
    #         t=t,
    #         format_labels=format_labels,
    #     )

    # def set_supxyt(
    #     self,
    #     xlabel=None,
    #     ylabel=None,
    #     title=None,
    #     format_labels=True,
    # ):
    #     self.axis = ax_module.set_supxyt(
    #         self.axis,
    #         xlabel=xlabel,
    #         ylabel=ylabel,
    #         title=title,
    #         format_labels=format_labels,
    #     )

    # def set_ticks(
    #     self,
    #     xvals=None,
    #     xticks=None,
    #     yvals=None,
    #     yticks=None,
    #     **kwargs,
    # ):

    #     self.axis = ax_module.set_ticks(
    #         self.axis,
    #         xvals=xvals,
    #         xticks=xticks,
    #         yvals=yvals,
    #         yticks=yticks,
    #     )

    # def set_n_ticks(self, n_xticks=4, n_yticks=4):
    #     self.axis = ax_module.set_n_ticks(
    #         self.axis, n_xticks=n_xticks, n_yticks=n_yticks
    #     )

    # def hide_spines(
    #     self,
    #     top=True,
    #     bottom=True,
    #     left=True,
    #     right=True,
    #     ticks=True,
    #     labels=True,
    # ):
    #     self.axis = ax_module.hide_spines(
    #         self.axis,
    #         top=top,
    #         bottom=bottom,
    #         left=left,
    #         right=right,
    #         ticks=ticks,
    #         labels=labels,
    #     )

    # def extend(self, x_ratio=1.0, y_ratio=1.0):
    #     self.axis = ax_module.extend(
    #         self.axis, x_ratio=x_ratio, y_ratio=y_ratio
    #     )

    # def shift(self, dx=0, dy=0):
    #     self.axis = ax_module.shift(self.axis, dx=dx, dy=dy)


# EOF
