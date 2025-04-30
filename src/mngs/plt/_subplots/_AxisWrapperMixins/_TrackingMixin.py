#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 18:40:59 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/_TrackingMixin.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_AxisWrapperMixins/_TrackingMixin.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionality:
    * Handles tracking and history management for matplotlib plot operations
Input:
    * Plot method calls, their arguments, and tracking configuration
Output:
    * Tracked plotting history and DataFrame export for analysis
Prerequisites:
    * pandas, matplotlib
"""

from contextlib import contextmanager

import pandas as pd

from .._export_as_csv import export_as_csv as _export_as_csv


class TrackingMixin:
    """Mixin class for tracking matplotlib plotting operations.

    Example
    -------
    >>> fig, ax = plt.subplots()
    >>> ax.track = True
    >>> ax.id = 0
    >>> ax._ax_history = OrderedDict()
    >>> ax.plot([1, 2, 3], [4, 5, 6], id="plot1")
    >>> print(ax.history)
    {'plot1': ('plot1', 'plot', ([1, 2, 3], [4, 5, 6]), {})}
    """

    def _track(self, track, id, method_name, args, kwargs):
        # Extract id from kwargs and remove it before passing to matplotlib
        if hasattr(kwargs, "get") and "id" in kwargs:
            id = kwargs.pop("id")

        if track is None:
            track = self.track

        if track:
            id = id if id is not None else self.id
            self.id += 1
            self._ax_history[id] = (id, method_name, args, kwargs)

    @contextmanager
    def _no_tracking(self):
        """Context manager to temporarily disable tracking."""
        original_track = self.track
        self.track = False
        try:
            yield
        finally:
            self.track = original_track

    @property
    def history(self):
        return {k: self._ax_history[k] for k in self._ax_history}

    @property
    def flat(self):
        if isinstance(self._axis_mpl, list):
            return self._axis_mpl
        else:
            return [self._axis_mpl]

    def reset_history(self):
        self._ax_history = {}

    def export_as_csv(self):
        """
        Export tracked plotting data to a DataFrame in SigmaPlot format.
        """
        df = _export_as_csv(self.history)

        return df if df is not None else pd.DataFrame()

    # def _track(
    #     self,
    #     track: Optional[bool],
    #     plot_id: Optional[str],
    #     method_name: str,
    #     args: Any,
    #     kwargs: Dict[str, Any]
    # ) -> None:
    #     """Tracks plotting operation if tracking is enabled."""
    #     if track is None:
    #         track = self.track
    #     if track:
    #         plot_id = plot_id if plot_id is not None else self.id
    #         self.id += 1
    #         self._ax_history[plot_id] = (plot_id, method_name, args, kwargs)

    # @contextmanager
    # def _no_tracking(self) -> None:
    #     """Temporarily disables tracking within a context."""
    #     original_track = self.track
    #     self.track = False
    #     try:
    #         yield
    #     finally:
    #         self.track = original_track

    # @property
    # def history(self) -> Dict[str, Tuple]:
    #     """Returns the plotting history."""
    #     return dict(self._ax_history)

    # def reset_history(self) -> None:
    #     """Clears the plotting history."""
    #     self._ax_history = OrderedDict()

    # def export_as_csv(self) -> pd.DataFrame:
    #     """Converts plotting history to a SigmaPlot-compatible DataFrame."""
    #     df = _export_as_csv(self.history)
    #     return df if df is not None else pd.DataFrame()

# EOF