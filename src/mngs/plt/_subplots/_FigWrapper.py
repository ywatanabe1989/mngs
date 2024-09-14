#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-09-14 17:17:11 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/_FigWrapper.py

from functools import wraps

import numpy as np
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
        # fig.tight_layout()
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
    # def legend(self, loc="upper left"):
    #     for ax in self.axes:
    #         ax.legend(loc=loc)

    def to_sigma(self):
        """
        Summarizes all data under the figure, including all AxesWrapper objects.

        Returns
        -------
        pd.DataFrame
            Concatenated dataframe of all axes data with spacer columns.

        Example
        -------
        fig, axes = mngs.plt.subplots(2, 2)
        df_summary = fig.to_sigma()
        print(df_summary)
        """
        dfs = []
        for i_ax, ax in enumerate(self.axes):
            if hasattr(ax, "to_sigma"):
                df = ax.to_sigma()
                if not df.empty:
                    df.columns = [f"ax_{i_ax:02d}_{col}" for col in df.columns]
                    dfs.append(df)

                    # Add a spacer column after each non-empty dataframe except the last one
                    if i_ax < len(self.axes) - 1:
                        spacer = pd.DataFrame({"Spacer": [np.nan] * len(df)})
                        dfs.append(spacer)

        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

        # dfs = []
        # for i_ax, ax in enumerate(self.axes):
        #     if hasattr(ax, "to_sigma"):
        #         try:
        #             df = ax.to_sigma()
        #             df.columns = [f"ax_{i_ax:02d}_{col}" for col in df.columns]
        #             dfs.append(df)
        #         except Exception as e:
        #             print(e)
        #             __import__("ipdb").set_trace()

        #     # Add a spacer column after each dataframe except the last one
        #     if i_ax < len(self.axes) - 1:
        #         spacer = pd.DataFrame({"Spacer": [np.nan] * len(df)})
        #         dfs.append(spacer)

        # if dfs:
        #     return pd.concat(dfs, axis=1)
        # else:
        #     return pd.DataFrame()

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
