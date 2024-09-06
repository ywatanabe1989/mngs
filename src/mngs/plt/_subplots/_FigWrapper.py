#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-30 01:43:44 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/_FigWrapper.py

from functools import wraps

import pandas as pd
from mngs.gen import deprecated


class FigWrapper:
    """
    A wrapper class for a Matplotlib axis that collects plotting data.
    """

    def __init__(self, fig):
        """
        Initialize the AxisWrapper with a given axis and history reference.
        """
        self.fig = fig
        self.axes = []

    def __getattr__(self, attr):
        """
        Wrap the axis attribute access to collect plot calls or return the attribute directly.
        """
        original_attr = getattr(self.fig, attr)

        if callable(original_attr):

            @wraps(original_attr)
            def wrapper(*args, track=None, id=None, **kwargs):
                results = original_attr(*args, **kwargs)
                self._track(track, id, attr, args, kwargs)
                return results

            return wrapper

        else:
            return original_attr

    ################################################################################
    # Original methods
    ################################################################################
    def to_sigma(self):
        """
        Summarizes all data under the figure, including all AxesWrapper objects.
        """
        dfs = []
        for i, ax in enumerate(self.axes):
            if hasattr(ax, "to_sigma"):
                df = ax.to_sigma()
                df.columns = [f"Axis_{i}_{col}" for col in df.columns]
                dfs.append(df)

        if dfs:
            return pd.concat(dfs, axis=1)
        else:
            return pd.DataFrame()

    @deprecated("Use supxyt() instead.")
    def set_supxyt(self, *args, **kwargs):
        return self.supxyt(*args, **kwargs)

    def supxyt(self, x=False, y=False, t=False):
        """Sets xlabel, ylabel and title"""
        if x is not False:
            self.fig.supxlabel(x)
        if y is not False:
            self.fig.supylabel(y)
        if t is not False:
            self.fig.suptitle(t)
        return self.fig

    def tight_layout(self, rect=[0, 0.03, 1, 0.95]):
        self.fig.tight_layout(rect=rect)
