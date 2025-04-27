#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 11:54:36 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_AxisWrapper.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_AxisWrapper.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-13 14:53:28 (ywatanabe)"
# File: ./mngs_repo/src/mngs/plt/_subplots/_AxisWrapper.py

import warnings
from collections import OrderedDict
from functools import wraps

from ._AxisWrapperMixins import (
    AdjustmentMixin,
    BasicPlotMixin,
    SeabornMixin,
    TrackingMixin,
)


class AxisWrapper(
    BasicPlotMixin, SeabornMixin, AdjustmentMixin, TrackingMixin
):
    """Wrapper class for matplotlib axis with additional functionality."""

    def __init__(self, fig, axis, track):
        """Initialize the axis wrapper.

        Parameters
        ----------
        axis : Union[Axes, np.ndarray]
            Matplotlib axis or array of axes
        track : bool, optional
            Whether to track plotting operations, by default True
        """
        self.fig = fig
        self.axis = axis
        self._ax_history = OrderedDict()
        self.track = track
        self.id = 0

    def get_figure(
        self,
    ):
        return self.fig

    # def __getattr__(self, attr):
    #     if hasattr(self.axis, attr):
    #         original_attr = getattr(self.axis, attr)

    #         if callable(original_attr):

    #             @wraps(original_attr)
    #             def wrapper(*args, track=None, id=None, **kwargs):
    #                 results = original_attr(*args, **kwargs)
    #                 self._track(track, id, attr, args, kwargs)
    #                 return results

    #             return wrapper
    #         else:
    #             return original_attr
    #     raise AttributeError(
    #         f"'{type(self).__name__}' object has no attribute '{attr}'"
    #     )

    def __getattr__(self, attr):
        if hasattr(self.axis, attr):
            original = getattr(self.axis, attr)
            if callable(original):

                @wraps(original)
                def wrapper(*args, **kwargs):
                    return original(*args, **kwargs)

                return wrapper
            return original
        warnings.warn(
            f"MNGS AxisWrapper: '{attr}' is not implemented on {type(self.axis)}, call ignored.",
            UserWarning,
        )

        def dummy(*args, **kwargs):
            return None

        return dummy

# EOF