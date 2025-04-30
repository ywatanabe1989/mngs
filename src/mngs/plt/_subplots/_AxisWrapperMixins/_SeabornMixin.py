#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 18:27:40 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/_SeabornMixin.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_AxisWrapperMixins/_SeabornMixin.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from functools import wraps

import mngs
import numpy as np
import pandas as pd
import seaborn as sns

from ....plt import ax as ax_module


def sns_copy_doc(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    wrapper.__doc__ = getattr(sns, func.__name__.split("sns_")[-1]).__doc__
    return wrapper


class SeabornMixin:

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

            self._axis_mpl = sns_plot_fn(ax=self._axis_mpl, *args, **kwargs)

        # Track the plot if required
        track_obj = track_obj if track_obj is not None else args
        self._track(track, id, method_name, track_obj, kwargs)

    def _sns_base_xyhue(
        self, method_name, *args, track=True, id=None, **kwargs
    ):
        """Formats data passed to sns functions with (data=data, x=x, y=y) keyword arguments"""
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

    def _sns_prepare_xyhue(
        self, data=None, x=None, y=None, hue=None, **kwargs
    ):
        """Returns obj to track"""
        data = data.reset_index()

        if hue is not None:
            if x is None and y is None:

                return data
            elif x is None:

                agg_dict = {}
                for hh in data[hue].unique():
                    agg_dict[hh] = data.loc[data[hue] == hh, y]
                df = mngs.pd.force_df(agg_dict)
                return df

            elif y is None:

                df = pd.concat(
                    [
                        data.loc[data[hue] == hh, x]
                        for hh in data[hue].unique()
                    ],
                    axis=1,
                )
                return df
            else:
                pivoted_data = data.pivot_table(
                    values=y,
                    index=data.index,
                    columns=[x, hue],
                    aggfunc="first",
                )
                pivoted_data.columns = [
                    f"{col[0]}-{col[1]}" for col in pivoted_data.columns
                ]
                return pivoted_data
        else:
            if x is None and y is None:
                return data

            elif x is None:
                return data[[y]]

            elif y is None:
                return data[[x]]

            else:
                return data.pivot_table(
                    values=y, index=data.index, columns=x, aggfunc="first"
                )

    @sns_copy_doc
    def sns_barplot(
        self, data=None, x=None, y=None, track=True, id=None, **kwargs
    ):
        self._sns_base_xyhue(
            "sns_barplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_boxplot(
        self,
        data=None,
        x=None,
        y=None,
        strip=False,
        track=True,
        id=None,
        **kwargs,
    ):
        self._sns_base_xyhue(
            "sns_boxplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )
        if strip:
            strip_kwargs = kwargs.copy()
            strip_kwargs.pop("notch", None)  # Remove boxplot-specific kwargs
            strip_kwargs.pop("whis", None)
            self.sns_stripplot(
                data=data,
                x=x,
                y=y,
                track=False,
                id=f"{id}_strip",
                **strip_kwargs,
            )

    @sns_copy_doc
    def sns_heatmap(self, *args, xyz=False, track=True, id=None, **kwargs):
        method_name = "sns_heatmap"
        df = args[0]
        if xyz:
            df = mngs.pd.to_xyz(df)
        self._sns_base(
            method_name, *args, track=track, track_obj=df, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_histplot(
        self, data=None, x=None, y=None, track=True, id=None, **kwargs
    ):
        self._sns_base_xyhue(
            "sns_histplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_kdeplot(
        self,
        data=None,
        x=None,
        y=None,
        xlim=None,
        ylim=None,
        track=True,
        id=None,
        **kwargs,
    ):
        if kwargs.get("hue"):
            hue_col = kwargs["hue"]
            hues = data[hue_col]
            if x is not None:
                lim = xlim
                for hh in np.unique(hues):
                    _data = data.loc[data[hue_col] == hh, x]
                    self.kde(_data, xlim=lim, label=hh, id=hh, **kwargs)

            if y is not None:
                lim = xlim
                for hh in np.unique(hues):
                    _data = data.loc[data[hue] == hh, y]
                    self.kde(_data, xlim=lim, label=hh, id=hh, **kwargs)

        else:
            if x is not None:
                _data, lim = data[x], xlim
            if y is not None:
                _data, lim = data[y], ylim
            self.kde(_data, xlim=lim, **kwargs)

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
    def sns_swarmplot(
        self, data=None, x=None, y=None, track=True, id=None, **kwargs
    ):
        self._sns_base_xyhue(
            "sns_swarmplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    @sns_copy_doc
    def sns_stripplot(
        self, data=None, x=None, y=None, track=True, id=None, **kwargs
    ):
        self._sns_base_xyhue(
            "sns_stripplot", data=data, x=x, y=y, track=track, id=id, **kwargs
        )

    # @sns_copy_doc
    # def sns_violinplot(
    #     self, data=None, x=None, y=None, track=True, id=None, **kwargs
    # ):
    #     self._sns_base_xyhue(
    #         "sns_violinplot", data=data, x=x, y=y, track=track, id=id, **kwargs
    #     )

    @sns_copy_doc
    def sns_violinplot(
        self,
        data=None,
        x=None,
        y=None,
        track=True,
        id=None,
        half=False,
        **kwargs,
    ):
        if half:
            with self._no_tracking():
                self._axis_mpl = ax_module.half_violin(
                    self._axis_mpl, data=data, x=x, y=y, **kwargs
                )
        else:
            self._sns_base_xyhue(
                "sns_violinplot",
                data=data,
                x=x,
                y=y,
                track=track,
                id=id,
                **kwargs,
            )

        # Tracking
        track_obj = self._sns_prepare_xyhue(data, x, y, kwargs.get("hue"))
        self._track(track, id, "sns_violinplot", track_obj, kwargs)

        return self._axis_mpl

    @sns_copy_doc
    def sns_jointplot(self, *args, track=True, id=None, **kwargs):
        self._sns_base("sns_jointplot", *args, track=track, id=id, **kwargs)

# EOF