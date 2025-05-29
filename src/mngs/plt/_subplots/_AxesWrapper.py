#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 15:04:11 (ywatanabe)"
# File: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_AxesWrapper.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_AxesWrapper.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from functools import wraps

import pandas as pd


class AxesWrapper:
    def __init__(self, fig_mngs, axes_mngs):
        self._fig_mngs = fig_mngs
        self._axes_mngs = axes_mngs

    def get_figure(self):
        return self._fig_mngs

    def __dir__(self):
        # Combine attributes from both self and the wrapped matplotlib axes
        attrs = set(dir(self.__class__))
        attrs.update(object.__dir__(self))

        # Add attributes from the axes objects if available
        if hasattr(self, "_axes_mngs") and self._axes_mngs is not None:
            # Get attributes from the first axis if there are any
            if self._axes_mngs.size > 0:
                first_ax = self._axes_mngs.flat[0]
                attrs.update(dir(first_ax))

        return sorted(attrs)

    # def __dir__(self):
    #     # Combine attributes from both self and the wrapped matplotlib axes
    #     attrs = set(dir(self.__class__))
    #     attrs.update(object.__dir__(self))
    #     attrs.update(dir(self._axes_mpl))
    #     return sorted(attrs)

    # def __getattr__(self, attr):
    #     # print(f"Attribute of FigWrapper: {attr}")
    #     attr_mpl = getattr(self._axes_mngs, attr)

    #     if callable(attr_mpl):

    #         @wraps(attr_mpl)
    #         def wrapper(*args, track=None, id=None, **kwargs):
    #             results = attr_mpl(*args, **kwargs)
    #             # self._track(track, id, attr, args, kwargs)
    #             return results

    #         return wrapper

    #     else:
    #         return attr_mpl

    def __getattr__(self, name):
        # Note that self._axes_mngs is "numpy.ndarray"
        # print(f"Attribute of AxesWrapper: {name}")
        methods = []
        try:
            for axis in self._axes_mngs.flat:
                methods.append(getattr(axis, name))
        except Exception:
            methods = []

        if methods and all(callable(m) for m in methods):

            @wraps(methods[0])
            def wrapper(*args, **kwargs):
                return [
                    getattr(ax, name)(*args, **kwargs)
                    for ax in self._axes_mngs.flat
                ]

            return wrapper

        if methods and not callable(methods[0]):
            return methods

        # warnings.warn(
        #     f"MNGS AxesWrapper: '{name}' not implemented, ignored.",
        #     UserWarning,
        # )

        def dummy(*args, **kwargs):
            return None

        return dummy

    def __getitem__(self, index):
        subset = self._axes_mngs[index]
        # Handle slice or numpy array result (when accessing row/column)
        if isinstance(subset, (slice, type(self._axes_mngs))):
            if hasattr(subset, 'ndim') and subset.ndim > 0:
                return AxesWrapper(self._fig_mngs, subset)
        return subset

    def __iter__(self):
        return iter(self._axes_mngs)

    def __len__(self):
        return self._axes_mngs.size

    def legend(self, loc="upper left"):
        return [ax.legend(loc=loc) for ax in self._axes_mngs.flat]

    @property
    def history(self):
        return [ax.history for ax in self._axes_mngs.flat]

    @property
    def shape(self):
        return self._axes_mngs.shape
    
    @property
    def flat(self):
        """Return a flattened iterator over all axes, mimicking numpy behavior."""
        return self._axes_mngs.flat

    def export_as_csv(self):
        dfs = []
        for ii, ax in enumerate(self._axes_mngs.flat):
            df = ax.export_as_csv()
            df.columns = [f"ax_{ii:02d}_{col}" for col in df.columns]
            dfs.append(df)
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

# EOF