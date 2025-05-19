#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 17:11:28 (ywatanabe)"
# File: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_AxisWrapper.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_AxisWrapper.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import warnings
from functools import wraps

import matplotlib

from ._AxisWrapperMixins import (AdjustmentMixin, MatplotlibPlotMixin,
                                 SeabornMixin, TrackingMixin)


class AxisWrapper(
    MatplotlibPlotMixin, SeabornMixin, AdjustmentMixin, TrackingMixin
):
    def __init__(self, fig_mngs, axis_mpl, track):
        """Initialize the AxisWrapper.
        
        Args:
            fig_mngs: Parent figure wrapper
            axis_mpl: Matplotlib axis to wrap
            track: Whether to track plotting operations
        """
        self._fig_mpl = fig_mngs._fig_mpl
        # Axis Properties
        # self.axis = axis_mpl
        # self._axis = axis_mpl
        # self._axis_mngs = self
        self._axis_mpl = axis_mpl

        # Axes Properties
        # self.axes = axis_mpl
        # self._axes = axis_mpl
        self._axes_mpl = axis_mpl
        # self._axes_mngs = self

        # Tracking properties
        self._ax_history = {}
        self._method_counters = {}  # Track method counts for auto-generated IDs
        self.track = track
        self.id = 0
        self._counter_part = matplotlib.axes.Axes

    def get_figure(self):
        return self._fig_mpl

    def __getattr__(self, name):
        # 0. Check if the attribute is explicitly defined in AxisWrapper or its Mixins
        #    This check happens implicitly before __getattr__ is called.
        #    If a method like `plot` is defined in BasicPlotMixin, it will be found first.

        # print(f"Attribute of AxisWrapper: {name}")

        # 1. Try to get the attribute from the wrapped axes instance
        if hasattr(self._axes_mpl, name):
            orig_attr = getattr(self._axes_mpl, name)

            if callable(orig_attr):

                @wraps(orig_attr)
                def wrapper(*args, **kwargs):
                    id_value = kwargs.pop("id", None)
                    track_override = kwargs.pop("track", None)

                    # Call the original matplotlib method
                    result = orig_attr(*args, **kwargs)

                    # Determine if tracking should occur
                    should_track = (
                        track_override
                        if track_override is not None
                        else self.track
                    )

                    # Track the method call if tracking enabled for this call
                    # We only track if an 'id' was provided, explicit tracking methods handle other cases
                    if should_track and id_value is not None:
                        # Use the _track method from TrackingMixin
                        # Pass method name, args, kwargs (original ones, maybe without id/track?)
                        # The current _track implementation in the mixin needs review for consistency
                        # Let's assume _track handles getting the method name and uses id_value
                        # For simplicity, just call the original method for now. Tracking needs refinement.
                        # --- Refined Tracking Call (assuming _track exists and works) ---
                        try:
                            # Convert args to tracked_dict for consistency with other tracking
                            tracked_dict = {"args": args}
                            self._track(
                                should_track, id_value, name, tracked_dict, kwargs
                            )
                        except AttributeError:
                            warnings.warn(
                                f"Tracking setup incomplete for AxisWrapper ({name}).",
                                UserWarning,
                                stacklevel=2,
                            )
                        # ------------------------------------------------------------
                    return result  # Return the result of the original call

                return wrapper
            else:
                # If it's a non-callable attribute (property, etc.), return it directly
                return orig_attr

        # 2. If not found on instance, try the counterpart type (fallback)
        if hasattr(self._counter_part, name):
            counterpart_attr = getattr(self._counter_part, name)
            warnings.warn(
                f"MNGS Axis_MplWrapper: '{name}' not directly handled. "
                f"Falling back to underlying '{self._counter_part.__name__}' attribute.",
                UserWarning,
                stacklevel=2,
            )
            # If the counterpart attribute is callable (likely a method descriptor)
            if callable(counterpart_attr):
                # Return a new function that calls the counterpart method on self._axes_mpl
                @wraps(counterpart_attr)
                def fallback_method(*args, **kwargs):
                    # Note: No id/track handling for fallback methods
                    return counterpart_attr(self._axes_mpl, *args, **kwargs)

                return fallback_method
            else:
                # Non-callable class attribute. Attempt to get from instance again,
                # otherwise return the class attribute/descriptor.
                try:
                    return getattr(self._axes_mpl, name)
                except AttributeError:
                    return counterpart_attr

        # 3. If not found anywhere, raise AttributeError
        raise AttributeError(
            f"'{type(self).__name__}' object and its underlying '{self._counter_part.__name__}' "
            f"have no attribute '{name}'"
        )

    def __dir__(self):
        # Combine attributes from both self and the wrapped matplotlib figure
        attrs = set(dir(self.__class__))
        attrs.update(object.__dir__(self))
        attrs.update(dir(self._axes_mpl))
        return sorted(attrs)
        
    def flatten(self):
        """Return a list containing just this axis.
        
        This method makes AxisWrapper compatible with code that calls flatten()
        on an axes collection. It returns a list containing just this single axis
        to maintain consistency with AxesWrapper.flatten().
        
        Returns:
            list: A list containing this axis wrapper
            
        Example:
            # When working with either AxesWrapper or AxisWrapper, this works:
            axes_list = list(axes.flatten())
        """
        return [self]


"""
import matplotlib.pyplot as plt
import mngs.plt as mplt

fig_mngs, axes = plt.subplots(ncols=2)
mfig_mngs, maxes = mplt.subplots(ncols=2)

print(set(dir(mfig_mngs)) - set(dir(fig_mngs)))
print(set(dir(maxes)) - set(dir(axes)))

is_compatible = np.all([kk in set(dir(msubplots)) for kk in set(dir(counter_part))])
if is_compatible:
    print(f"{msubplots.__name__} is compatible with {counter_part.__name__}")
else:
    print(f"{msubplots.__name__} is incompatible with {counter_part.__name__}")
"""

# EOF