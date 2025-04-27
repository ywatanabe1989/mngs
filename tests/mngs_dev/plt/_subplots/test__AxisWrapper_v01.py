# --------------------------------------------------------------------------------
# Start of Source code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/_subplots/_AxisWrapper_v01.py
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
#
# --------------------------------------------------------------------------------
# End of Source code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/_subplots/_AxisWrapper_v01.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
