#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 20:27:20 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_AxisWrapper.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_AxisWrapper.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import warnings

from ._AxisWrapperMixins import (
    AdjustmentMixin,
    BasicPlotMixin,
    SeabornMixin,
    TrackingMixin,
)


class AxisWrapper(
    BasicPlotMixin, SeabornMixin, AdjustmentMixin, TrackingMixin
):
    def __init__(self, fig, axis, track):
        self.fig = fig
        self.axis = axis
        self._ax_history = {}
        self.track = track
        self.id = 0

    def get_figure(self):
        return self.fig

    def __getattr__(self, attr):
        if hasattr(self.axis, attr):
            original = getattr(self.axis, attr)
            if callable(original):

                # @wraps(original)
                # def wrapper(*args, **kwargs):
                #     result = original(*args, **kwargs)
                #     self._track(
                #         kwargs.get("track", None),
                #         kwargs.get("id", None),
                #         attr,
                #         args,
                #         kwargs,
                #     )
                #     return result

                # src/mngs/plt/_subplots/_AxisWrapper.py
                # Fix for the indentation error on lines 57-59
                def wrapper(*args, **kwargs):
                    # Handle 'id' parameter which is not supported by matplotlib
                    id_value = (
                        kwargs.pop("id", None)
                        if isinstance(kwargs, dict)
                        else None
                    )
                    result = original(*args, **kwargs)
                    if id_value is not None and self._track:
                        # Add tracking code here
                        pass  # Add a pass statement if there's no implementation yet
                    return result

            return original

        warnings.warn(
            f"MNGS AxisWrapper: '{attr}' not implemented, ignored.",
            UserWarning,
        )

        def dummy(*args, **kwargs):
            return None

        return dummy

# EOF