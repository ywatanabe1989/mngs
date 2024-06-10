#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-07 19:18:21 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/FigWrapper.py

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

    def __getattr__(self, attr):
        """
        Wrap the axis attribute access to collect plot calls or return the attribute directly.
        """
        original_attr = getattr(self.fig, attr)

        if callable(original_attr):

            def wrapper(
                *args, id=None, track=True, n_xticks=4, n_yticks=4, **kwargs
            ):
                result = original_attr(*args, **kwargs)

                return result

            return wrapper
        else:

            return original_attr

    ################################################################################
    # Original methods
    ################################################################################
    @deprecated("Use supxyt() instead.")
    def set_supxyt(self, *args, **kwargs):
        return self.supxyt(*args, **kwargs)

    def supxyt(self, xlabel=None, ylabel=None, title=None):
        """Sets xlabel, ylabel and title"""
        if xlabel is not None:
            self.fig.supxlabel(xlabel)
        if ylabel is not None:
            self.fig.supylabel(ylabel)
        if title is not None:
            self.fig.suptitle(title)
        return self.fig

    def tight_layout(self, rect=[0, 0.03, 1, 0.95]):
        self.fig.tight_layout(rect=rect)
