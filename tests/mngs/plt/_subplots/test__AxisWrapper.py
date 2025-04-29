#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:48:28 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/_subplots/test__AxisWrapper.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/_subplots/test__AxisWrapper.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
from unittest.mock import MagicMock

import pytest
from mngs.plt._subplots._AxisWrapper import AxisWrapper


class TestAxisWrapper:
    def setup_method(self):
        self.fig_mock = MagicMock()
        self.axis_mock = MagicMock()
        self.wrapper = AxisWrapper(self.fig_mock, self.axis_mock, track=True)

    def test_init(self):
        assert self.wrapper.fig is self.fig_mock
        assert self.wrapper.axis is self.axis_mock
        assert self.wrapper._ax_history == {}
        assert self.wrapper.track is True
        assert self.wrapper.id == 0

    def test_get_figure(self):
        assert self.wrapper.get_figure() is self.fig_mock

    def test_getattr_existing_attribute(self):
        # Test accessing an existing attribute on the axis
        self.axis_mock.get_xlim = lambda: (0, 1)
        assert self.wrapper.get_xlim() == (0, 1)

    def test_getattr_warning(self):
        # Test attempting to access a non-existent attribute
        with pytest.warns(UserWarning, match="not implemented, ignored"):
            result = self.wrapper.nonexistent_method()
            assert result is None

    def test_function_with_id_parameter(self):
        # Test that id parameter is handled correctly
        self.axis_mock.plot = MagicMock(return_value="plot_result")

        # Call plot with id
        result = self.wrapper.plot([1, 2, 3], [4, 5, 6], id="test_plot")

        # Check that plot was called without the id parameter
        self.axis_mock.plot.assert_called_once()
        args, kwargs = self.axis_mock.plot.call_args
        assert "id" not in kwargs

        # And the result should be what the original method returned
        assert result == "plot_result"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_AxisWrapper.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-27 20:27:20 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_AxisWrapper.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_subplots/_AxisWrapper.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import warnings
# 
# from ._AxisWrapperMixins import (
#     AdjustmentMixin,
#     BasicPlotMixin,
#     SeabornMixin,
#     TrackingMixin,
# )
# 
# 
# class AxisWrapper(
#     BasicPlotMixin, SeabornMixin, AdjustmentMixin, TrackingMixin
# ):
#     def __init__(self, fig, axis, track):
#         self.fig = fig
#         self.axis = axis
#         self._ax_history = {}
#         self.track = track
#         self.id = 0
# 
#     def get_figure(self):
#         return self.fig
# 
#     def __getattr__(self, attr):
#         if hasattr(self.axis, attr):
#             original = getattr(self.axis, attr)
#             if callable(original):
# 
#                 # @wraps(original)
#                 # def wrapper(*args, **kwargs):
#                 #     result = original(*args, **kwargs)
#                 #     self._track(
#                 #         kwargs.get("track", None),
#                 #         kwargs.get("id", None),
#                 #         attr,
#                 #         args,
#                 #         kwargs,
#                 #     )
#                 #     return result
# 
#                 # src/mngs/plt/_subplots/_AxisWrapper.py
#                 # Fix for the indentation error on lines 57-59
#                 def wrapper(*args, **kwargs):
#                     # Handle 'id' parameter which is not supported by matplotlib
#                     id_value = (
#                         kwargs.pop("id", None)
#                         if isinstance(kwargs, dict)
#                         else None
#                     )
#                     result = original(*args, **kwargs)
#                     if id_value is not None and self._track:
#                         # Add tracking code here
#                         pass  # Add a pass statement if there's no implementation yet
#                     return result
# 
#             return original
# 
#         warnings.warn(
#             f"MNGS AxisWrapper: '{attr}' not implemented, ignored.",
#             UserWarning,
#         )
# 
#         def dummy(*args, **kwargs):
#             return None
# 
#         return dummy
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_AxisWrapper.py
# --------------------------------------------------------------------------------
