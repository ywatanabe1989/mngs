#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 19:52:56 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_AxesWrapper.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_AxesWrapper.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-13 14:53:28 (ywatanabe)"
# File: ./mngs_repo/src/mngs/plt/_subplots/_AxisWrapper.py

import warnings
from functools import wraps

import pandas as pd


class AxesWrapper:
    def __init__(self, fig, axes):
        self.fig = fig
        self.axes = axes

    def get_figure(self):
        return self.fig

    def __getattr__(self, name):
        methods = []
        try:
            for axis in self.axes.flat:
                methods.append(getattr(axis, name))
        except Exception:
            methods = []

        if methods and all(callable(m) for m in methods):

            @wraps(methods[0])
            def wrapper(*args, **kwargs):
                return [
                    getattr(ax, name)(*args, **kwargs) for ax in self.axes.flat
                ]

            return wrapper

        if methods and not callable(methods[0]):
            return methods

        warnings.warn(
            f"MNGS AxesWrapper: '{name}' not implemented, ignored.",
            UserWarning,
        )

        def dummy(*args, **kwargs):
            return None

        return dummy

    def __getitem__(self, index):
        subset = self.axes[index]
        if isinstance(index, slice):
            return AxesWrapper(self.fig, subset)
        return subset

    def __iter__(self):
        return iter(self.axes.flat)

    def __len__(self):
        return self.axes.size

    def legend(self, loc="upper left"):
        return [ax.legend(loc=loc) for ax in self.axes.flat]

    @property
    def history(self):
        return [ax.history for ax in self.axes.flat]

    @property
    def shape(self):
        return self.axes.shape

    def to_sigma(self):
        dfs = []
        for ii, ax in enumerate(self.axes.flat):
            df = ax.to_sigma()
            df.columns = [f"ax_{ii:02d}_{col}" for col in df.columns]
            dfs.append(df)
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

# EOF