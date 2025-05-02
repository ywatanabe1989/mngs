#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 17:11:51 (ywatanabe)"
# File: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_FigWrapper.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_FigWrapper.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from functools import wraps

import pandas as pd


class FigWrapper:
    def __init__(self, fig_mpl):
        self._fig_mpl = fig_mpl
        self._last_saved_info = None
        self._not_saved_yet_flag = True
        self._called_from_mng_io_save = False

    @property
    def figure(
        self,
    ):
        return self._fig_mpl

    def __getattr__(self, attr):
        # print(f"Attribute of FigWrapper: {attr}")
        attr_mpl = getattr(self._fig_mpl, attr)

        if callable(attr_mpl):

            @wraps(attr_mpl)
            def wrapper(*args, track=None, id=None, **kwargs):
                results = attr_mpl(*args, **kwargs)
                # self._track(track, id, attr, args, kwargs)
                return results

            return wrapper

        else:
            return attr_mpl

    # def __dir__(self):
    #     # Combine attributes from both self and the wrapped matplotlib figure
    #     attrs = set(dir(self.__class__))
    #     attrs.update(object.__dir__(self))
    #     attrs.update(dir(self._fig_mpl))
    #     return sorted(attrs)

    # def savefig(self, fname, *args, **kwargs):
    #     if not self._called_from_mng_io_save:
    #         warnings.warn(
    #             f"Instead of `FigWrapper.savefig({fname})`, use `mngs.io.save(fig, {fname}, symlink_from_cwd=True)` to handle symlink and export as csv.",
    #             UserWarning,
    #         )
    #         self._called_from_mng_io_save = False
    #     self._fig_mpl.savefig(fname, *args, **kwargs)

    def export_as_csv(self):
        """Export plotted data from all axes."""
        dfs = []
        for ii, ax in enumerate(self.axes.flat):
            if hasattr(ax, "export_as_csv"):
                df = ax.export_as_csv()
                if not df.empty:
                    df.columns = [f"ax_{ii:02d}_{col}" for col in df.columns]
                    dfs.append(df)

        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    def legend(self, *args, loc="upper left", **kwargs):
        """Legend with upper left by default."""
        for ax in self.axes:
            try:
                ax.legend(*args, loc=loc, **kwargs)
            except:
                pass

    def supxyt(self, x=False, y=False, t=False):
        """Wrapper for supxlabel, supylabel, and suptitle"""
        if x is not False:
            self._fig_mpl.supxlabel(x)
        if y is not False:
            self._fig_mpl.supylabel(y)
        if t is not False:
            self._fig_mpl.suptitle(t)
        return self._fig_mpl

    def tight_layout(self, *, rect=[0, 0.03, 1, 0.95], **kwargs):
        """Wrapper for tight_layout with rect=[0, 0.03, 1, 0.95] by default"""
        self._fig_mpl.tight_layout(rect=rect, **kwargs)

# EOF