#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 16:39:21 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/_subplots/test__AxisWrapper.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/_subplots/test__AxisWrapper.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/_subplots/_AxisWrapper.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-13 14:53:28 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/plt/_subplots/_AxisWrapper.py
#
# from collections import OrderedDict
# from functools import wraps
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
#     """Wrapper class for matplotlib axis with additional functionality."""
#
#     def __init__(self, fig, axis, track):
#         """Initialize the axis wrapper.
#
#         Parameters
#         ----------
#         axis : Union[Axes, np.ndarray]
#             Matplotlib axis or array of axes
#         track : bool, optional
#             Whether to track plotting operations, by default True
#         """
#         self.fig = fig
#         self.axis = axis
#         self._ax_history = OrderedDict()
#         self.track = track
#         self.id = 0
#
#     def get_figure(
#         self,
#     ):
#         return self.fig
#
#     def __getattr__(self, attr):
#         if hasattr(self.axis, attr):
#             original_attr = getattr(self.axis, attr)
#
#             if callable(original_attr):
#
#                 @wraps(original_attr)
#                 def wrapper(*args, track=None, id=None, **kwargs):
#                     results = original_attr(*args, **kwargs)
#                     self._track(track, id, attr, args, kwargs)
#                     return results
#
#                 return wrapper
#             else:
#                 return original_attr
#         raise AttributeError(
#             f"'{type(self).__name__}' object has no attribute '{attr}'"
#         )
#
#
# # EOF

import sys

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

# tests/mngs/plt/_subplots/test__AxisWrapper.py

project_root = Path(__file__).resolve().parents[3]
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mngs.plt._subplots._AxisWrapper import AxisWrapper


class Dummy:
    def foo(self, *args, **kwargs):
        return "bar"


def test_axiswrapper_getattr_and_error():
    dummy = Dummy()
    wrapper = AxisWrapper("fig", dummy, track=True)

    # existing method should forward call
    assert wrapper.foo(1, 2) == "bar"

    # unknown attr should raise
    with pytest.raises(AttributeError):
        _ = wrapper.nonexistent

# EOF