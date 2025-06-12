#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 14:56:48 (ywatanabe)"
# File: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_FigWrapper.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/plt/_subplots/_FigWrapper.py"
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

    def __dir__(self):
        # Combine attributes from both self and the wrapped matplotlib figure
        attrs = set(dir(self.__class__))
        attrs.update(object.__dir__(self))
        attrs.update(dir(self._fig_mpl))
        return sorted(attrs)

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
    
    def export_as_csv_for_sigmaplot(self, include_visual_params=True):
        """Export plotted data from all axes in SigmaPlot format.
        
        Parameters
        ----------
        include_visual_params : bool, optional
            Whether to include visual parameters at the top of the CSV.
            Default is True.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the plotted data formatted for SigmaPlot.
        """
        # If figure has multiple axes, concatenate their exports
        dfs = []
        for ii, ax in enumerate(self.axes.flat):
            if hasattr(ax, "export_as_csv_for_sigmaplot"):
                df = ax.export_as_csv_for_sigmaplot(include_visual_params=include_visual_params)
                if not df.empty:
                    # For multiple axes, we only include visual params once
                    if ii > 0 and include_visual_params:
                        # Remove visual parameter columns from subsequent axes
                        cols_to_keep = [col for col in df.columns 
                                       if col not in ["visual parameter label", "visual parameter value", "xticks", "yticks"]
                                       and not col.startswith("preserved")]
                        df = df[cols_to_keep]
                    # Prefix columns with axis index
                    rename_dict = {}
                    for col in df.columns:
                        if col not in ["visual parameter label", "visual parameter value", "xticks", "yticks"] and not col.startswith("preserved"):
                            rename_dict[col] = f"ax_{ii:02d}_{col}"
                    df = df.rename(columns=rename_dict)
                    dfs.append(df)
        
        if dfs:
            # Concatenate all dataframes
            result = dfs[0]
            for df in dfs[1:]:
                # Ensure same number of rows by padding
                max_rows = max(len(result), len(df))
                if len(result) < max_rows:
                    padding = pd.DataFrame(index=range(len(result), max_rows))
                    result = pd.concat([result, padding], ignore_index=True)
                if len(df) < max_rows:
                    padding = pd.DataFrame(index=range(len(df), max_rows))
                    df = pd.concat([df, padding], ignore_index=True)
                # Concatenate horizontally
                result = pd.concat([result, df], axis=1)
            return result
        else:
            return pd.DataFrame()

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
        """Wrapper for tight_layout with rect=[0, 0.03, 1, 0.95] by default.
        
        Handles cases where certain axes (like colorbars) are incompatible
        with tight_layout. If the figure is using constrained_layout, this
        method does nothing as constrained_layout handles spacing automatically.
        """
        import warnings
        
        # Check if figure is already using constrained_layout
        if hasattr(self._fig_mpl, 'get_constrained_layout') and self._fig_mpl.get_constrained_layout():
            # Figure is using constrained_layout, which handles colorbars better
            # No need to call tight_layout
            return
        
        try:
            with warnings.catch_warnings():
                # Suppress the specific warning about incompatible axes
                warnings.filterwarnings("ignore", 
                                      message="This figure includes Axes that are not compatible with tight_layout")
                self._fig_mpl.tight_layout(rect=rect, **kwargs)
        except Exception:
            # If tight_layout fails completely, try constrained_layout as fallback
            try:
                self._fig_mpl.set_constrained_layout(True)
                self._fig_mpl.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04)
            except Exception:
                # If both fail, do nothing - figure will use default layout
                pass

    def adjust_layout(self, **kwargs):
        """Adjust the constrained layout parameters.
        
        Parameters
        ----------
        w_pad : float, optional
            Width padding around axes (default: 0.05)
        h_pad : float, optional
            Height padding around axes (default: 0.05)
        wspace : float, optional
            Width space between subplots (default: 0.02)
        hspace : float, optional
            Height space between subplots (default: 0.02)
        rect : list of 4 floats, optional
            Rectangle in normalized figure coordinates to fit the whole layout
            [left, bottom, right, top] (default: [0, 0, 1, 1])
        """
        if hasattr(self._fig_mpl, 'get_constrained_layout') and self._fig_mpl.get_constrained_layout():
            # Update constrained layout parameters
            self._fig_mpl.set_constrained_layout_pads(**kwargs)
        else:
            # Fall back to tight_layout with rect parameter if provided
            if 'rect' in kwargs:
                self.tight_layout(rect=kwargs['rect'])
    
    def close(self):
        """Close the underlying matplotlib figure"""
        import matplotlib.pyplot as plt
        plt.close(self._fig_mpl)
    
    @property 
    def number(self):
        """Return the figure number for matplotlib.pyplot.close() compatibility"""
        return self._fig_mpl.number

    def __del__(self):
        """Cleanup when FigWrapper is deleted"""
        try:
            import matplotlib.pyplot as plt
            plt.close(self._fig_mpl)
        except:
            pass


# EOF
