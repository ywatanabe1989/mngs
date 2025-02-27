# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/ax/_set_ticks.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-06 00:09:46 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/plt/ax/_set_ticks.py
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-09-06 23:33:56 (ywatanabe)"
# 
# import matplotlib.pyplot as plt
# import mngs
# import numpy as np
# from ...types import is_listed_X
# from ...dict._to_str import to_str
# # fixme, quite slow
# def set_ticks(ax, xvals=None, xticks=None, yvals=None, yticks=None):
#     ax = set_x_ticks(ax, x_vals=xvals, x_ticks=xticks)
#     ax = set_y_ticks(ax, y_vals=yvals, y_ticks=yticks)
#     canvas_type = type(ax.figure.canvas).__name__
#     if "TkAgg" in canvas_type:
#         ax.get_figure().canvas.draw()  # Redraw the canvas once after making all updates
#     return ax
# 
# 
# def set_x_ticks(ax, x_vals=None, x_ticks=None):
#     """
#     Set custom tick labels on the x and y axes based on specified values and desired ticks.
# 
#     Parameters:
#     - ax: The axis object to modify.
#     - x_vals: Array of x-axis values.
#     - x_ticks: List of desired tick labels on the x-axis.
#     - y_vals: Array of y-axis values.
#     - y_ticks: List of desired tick labels on the y-axis.
# 
#     Example:
#         import matplotlib.pyplot as plt
#         import numpy as np
# 
#         fig, axes = plt.subplots(nrows=4)
#         x = np.linspace(0, 10, 100)
#         y = np.sin(x)
#         for ax in axes:
#             ax.plot(x, y)  # Plot a sine wave
# 
#         set_ticks(axes[0])  # Do nothing # OK
#         set_ticks(axes[1], x_vals=x+3) # OK
#         set_ticks(axes[2], x_ticks=[1,2])  # OK
#         set_ticks(axes[3], x_vals=x+3, x_ticks=[4,5])  # Auto-generate ticks across the range
#         fig.tight_layout()
#         plt.show()
#     """
# 
#     def _avoid_overlaps(values):
#         values = np.array(values)
#         if ("int" in str(values.dtype)) or ("float" in str(values.dtype)):
#             values = values.astype(float) + np.arange(len(values)) * 1e-5
#         return values
# 
#     def _set_x_vals(ax, x_vals):
#         x_vals = _avoid_overlaps(x_vals)
#         new_x_axis = np.linspace(*ax.get_xlim(), len(x_vals))
#         ax.set_xticks(new_x_axis)
#         ax.set_xticklabels([f"{xv}" for xv in x_vals])
#         return ax
# 
#     def _set_x_ticks(ax, x_ticks):
#         x_ticks = np.array(x_ticks)
#         if x_ticks.dtype.kind in ["U", "S", "O"]:  # If x_ticks are strings
#             ax.set_xticks(range(len(x_ticks)))
#             ax.set_xticklabels(x_ticks)
#         else:
#             x_vals = np.array(
#                 [
#                     label.get_text().replace("−", "-")
#                     for label in ax.get_xticklabels()
#                 ]
#             )
#             x_vals = x_vals.astype(float)
#             x_indi = np.argmin(
#                 np.array(
#                     np.abs(x_vals[:, np.newaxis] - x_ticks[np.newaxis, :])
#                 ),
#                 axis=0,
#             )
#             ax.set_xticks(ax.get_xticks()[x_indi])
#             ax.set_xticklabels([f"{xt}" for xt in x_ticks])
#         return ax
# 
#     x_vals_passed = x_vals is not None
#     x_ticks_passed = x_ticks is not None
# 
#     if is_listed_X(x_ticks, dict):
#         x_ticks = [to_str(xt, delimiter="\n") for xt in x_ticks]
# 
#     if (not x_vals_passed) and (not x_ticks_passed):
#         # Do nothing
#         pass
# 
#     elif x_vals_passed and (not x_ticks_passed):
#         # Replaces the x axis to x_vals
#         x_ticks = np.linspace(x_vals[0], x_vals[-1], 4)
#         ax = _set_x_vals(ax, x_ticks)
# 
#     elif (not x_vals_passed) and x_ticks_passed:
#         # Locates 'x_ticks' on the original x axis
#         ax.set_xticks(x_ticks)
# 
#     elif x_vals_passed and x_ticks_passed:
#         if x_vals == "auto":
#             x_vals = np.arange(len(x_ticks))
# 
#         # Replaces the original x axis to 'x_vals' and locates the 'x_ticks' on the new axis
#         ax = _set_x_vals(ax, x_vals)
#         ax = _set_x_ticks(ax, x_ticks)
# 
#     return ax
# 
# 
# def set_y_ticks(ax, y_vals=None, y_ticks=None):
#     """
#     Set custom tick labels on the y-axis based on specified values and desired ticks.
# 
#     Parameters:
#     - ax: The axis object to modify.
#     - y_vals: Array of y-axis values where ticks should be placed.
#     - y_ticks: List of labels for ticks on the y-axis.
# 
#     Example:
#         import matplotlib.pyplot as plt
#         import numpy as np
# 
#         fig, ax = plt.subplots()
#         x = np.linspace(0, 10, 100)
#         y = np.sin(x)
#         ax.plot(x, y)  # Plot a sine wave
# 
#         set_y_ticks(ax, y_vals=y, y_ticks=['Low', 'High'])  # Set custom y-axis ticks
#         plt.show()
#     """
# 
#     def _avoid_overlaps(values):
#         values = np.array(values)
#         if ("int" in str(values.dtype)) or ("float" in str(values.dtype)):
#             values = values.astype(float) + np.arange(len(values)) * 1e-5
#         return values
# 
#     def _set_y_vals(ax, y_vals):
#         y_vals = _avoid_overlaps(y_vals)
#         new_y_axis = np.linspace(*ax.get_ylim(), len(y_vals))
#         ax.set_yticks(new_y_axis)
#         ax.set_yticklabels([f"{yv:.2f}" for yv in y_vals])
#         return ax
# 
#     # def _set_y_ticks(ax, y_ticks):
#     #     y_ticks = np.array(y_ticks)
#     #     y_vals = np.array(
#     #         [
#     #             label.get_text().replace("−", "-")
#     #             for label in ax.get_yticklabels()
#     #         ]
#     #     )
#     #     y_vals = y_vals.astype(float)
#     #     y_indi = np.argmin(
#     #         np.array(np.abs(y_vals[:, np.newaxis] - y_ticks[np.newaxis, :])),
#     #         axis=0,
#     #     )
# 
#     #     # y_indi = [np.argmin(np.abs(y_vals - yt)) for yt in y_ticks]
#     #     ax.set_yticks(ax.get_yticks()[y_indi])
#     #     ax.set_yticklabels([f"{yt}" for yt in y_ticks])
#     #     return ax
#     def _set_y_ticks(ax, y_ticks):
#         y_ticks = np.array(y_ticks)
#         if y_ticks.dtype.kind in ["U", "S", "O"]:  # If y_ticks are strings
#             ax.set_yticks(range(len(y_ticks)))
#             ax.set_yticklabels(y_ticks)
#         else:
#             y_vals = np.array(
#                 [
#                     label.get_text().replace("−", "-")
#                     for label in ax.get_yticklabels()
#                 ]
#             )
#             y_vals = y_vals.astype(float)
#             y_indi = np.argmin(
#                 np.array(
#                     np.abs(y_vals[:, np.newaxis] - y_ticks[np.newaxis, :])
#                 ),
#                 axis=0,
#             )
#             ax.set_yticks(ax.get_yticks()[y_indi])
#             ax.set_yticklabels([f"{yt}" for yt in y_ticks])
#         return ax
# 
#     y_vals_passed = y_vals is not None
#     y_ticks_passed = y_ticks is not None
# 
#     if is_listed_X(y_ticks, dict):
#         y_ticks = [to_str(yt, delimiter="\n") for yt in y_ticks]
# 
#     if (not y_vals_passed) and (not y_ticks_passed):
#         # Do nothing
#         pass
# 
#     elif y_vals_passed and (not y_ticks_passed):
#         # Replaces the y axis to y_vals
#         ax = _set_y_vals(ax, y_vals)
# 
#     elif (not y_vals_passed) and y_ticks_passed:
#         # Locates 'y_ticks' on the original y axis
#         ax.set_yticks(y_ticks)
# 
#     elif y_vals_passed and y_ticks_passed:
#         # Replaces the original y axis to 'y_vals' and locates the 'y_ticks' on the new axis
#         if y_vals == "auto":
#             y_vals = np.arange(len(y_ticks))
# 
#         ax = _set_y_vals(ax, y_vals)
#         ax = _set_y_ticks(ax, y_ticks)
#     return ax
# 
# 
# if __name__ == "__main__":
#     import mngs
# 
#     xx, tt, fs = mngs.dsp.demo_sig()
#     pha, amp, freqs = mngs.dsp.wavelet(xx, fs)
# 
#     i_batch, i_ch = 0, 0
#     ff = freqs[i_batch, i_ch]
#     fig, ax = mngs.plt.subplots()
# 
#     ax.imshow2d(amp[i_batch, i_ch])
# 
#     ax = set_ticks(
#         ax,
#         x_vals=tt,
#         x_ticks=[0, 1, 2, 3, 4],
#         y_vals=ff,
#         y_ticks=[0, 128, 256],
#     )
# 
#     plt.show()
# 
# 
# # EOF

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.plt.ax._set_ticks import *

class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        raise NotImplementedError("Test not yet implemented")

    def test_edge_cases(self):
        # Edge case testing
        raise NotImplementedError("Test not yet implemented")

    def test_error_handling(self):
        # Error handling testing
        raise NotImplementedError("Test not yet implemented")

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
