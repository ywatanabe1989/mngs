#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-13 08:13:38 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/FigWrapper.py

from mngs.gen import deprecated
from functools import wraps


class FigWrapper:
    """
    A wrapper class for a Matplotlib axis that collects plotting data.
    """

    def __init__(self, fig):
        """
        Initialize the AxisWrapper with a given axis and history reference.
        """
        self.fig = fig

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
