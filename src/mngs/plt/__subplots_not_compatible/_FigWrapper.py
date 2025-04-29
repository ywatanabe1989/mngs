#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 12:19:10 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_FigWrapper_v03.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_FigWrapper_v03.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import warnings
from functools import wraps

import pandas as pd


class FigWrapper:
    def __init__(self, fig):
        self.fig = fig
        self.axes = []

    def __getattr__(self, name):
        if hasattr(self.fig, name):
            orig = getattr(self.fig, name)
            if callable(orig):

                @wraps(orig)
                def wrapper(*args, **kwargs):
                    return orig(*args, **kwargs)

                return wrapper
            return orig

        warnings.warn(
            f"MNGS FigWrapper: '{name}' not implemented, ignored.",
            UserWarning,
        )

        def dummy(*args, **kwargs):
            return None

        return dummy

    def legend(self, loc="upper left"):
        for ax in self.axes:
            try:
                ax.legend(loc=loc)
            except:
                pass

    def export_as_csv(self):
        dfs = []
        for ii, ax in enumerate(self.axes.flat):
            if hasattr(ax, "export_as_csv"):
                df = ax.export_as_csv()
                if not df.empty:
                    df.columns = [f"ax_{ii:02d}_{col}" for col in df.columns]
                    dfs.append(df)
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    def supxyt(self, x=False, y=False, t=False):
        if x is not False:
            self.fig.supxlabel(x)
        if y is not False:
            self.fig.supylabel(y)
        if t is not False:
            self.fig.suptitle(t)
        return self.fig

    def tight_layout(self, rect=[0, 0.03, 1, 0.95]):
        self.fig.tight_layout(rect=rect)

# EOF