# src from here --------------------------------------------------------------------------------
# import matplotlib
# import numpy as np
# 
# 
# class OOMFormatter(matplotlib.ticker.ScalarFormatter):
#     def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
#         self.order = order
#         self.fformat = fformat
#         matplotlib.ticker.ScalarFormatter.__init__(
#             self, useOffset=offset, useMathText=mathText
#         )
# 
#     def _set_order_of_magnitude(self):
#         self.orderOfMagnitude = self.order
# 
#     def _set_format(self, vmin=None, vmax=None):
#         self.format = self.fformat
#         if self._useMathText:
#             self.format = r"$\mathdefault{%s}$" % self.format
# 
# 
# def sci_note(ax, fformat="%1.1f", x=False, y=False, scilimits=(-3, 3)):
#     order_x = 0
#     order_y = 0
# 
#     if x:
#         order_x = np.floor(np.log10(np.max(np.abs(ax.get_xlim())) + 1e-5))
#         ax.xaxis.set_major_formatter(
#             OOMFormatter(order=int(order_x), fformat=fformat)
#         )
#         ax.ticklabel_format(axis="x", style="sci", scilimits=scilimits)
#         ax.xaxis.labelpad = -22
#         shift_x = (ax.get_xlim()[0] - ax.get_xlim()[1]) * 0.01
#         ax.xaxis.get_offset_text().set_position((shift_x, 0))
# 
#     if y:
#         order_y = np.floor(np.log10(np.max(np.abs(ax.get_ylim())) + 1e-5))
#         ax.yaxis.set_major_formatter(
#             OOMFormatter(order=int(order_y), fformat=fformat)
#         )
#         ax.ticklabel_format(axis="y", style="sci", scilimits=scilimits)
#         ax.yaxis.labelpad = -20
#         shift_y = (ax.get_ylim()[0] - ax.get_ylim()[1]) * 0.01
#         ax.yaxis.get_offset_text().set_position((0, shift_y))
# 
#     return ax
# 
# 
# # class OOMFormatter(matplotlib.ticker.ScalarFormatter):
# #     def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
# #         self.order = order
# #         self.fformat = fformat
# #         matplotlib.ticker.ScalarFormatter.__init__(
# #             self, useOffset=offset, useMathText=mathText
# #         )
# 
# #     def _set_order_of_magnitude(self):
# #         self.orderOfMagnitude = self.order
# 
# #     def _set_format(self, vmin=None, vmax=None):
# #         self.format = self.fformat
# #         if self._useMathText:
# #             self.format = r"$\mathdefault{%s}$" % self.format
# 
# 
# # def sci_note(ax, fformat="%1.1f", x=False, y=False, scilimits=(-3, 3)):
# #     order_x = 0
# #     order_y = 0
# 
# #     if x:
# #         order_x = np.floor(np.log10(np.max(np.abs(ax.get_xlim())) + 1e-5))
# #         ax.xaxis.set_major_formatter(
# #             OOMFormatter(order=int(order_x), fformat=fformat)
# #         )
# #         ax.ticklabel_format(axis="x", style="sci", scilimits=scilimits)
# 
# #     if y:
# #         order_y = np.floor(np.log10(np.max(np.abs(ax.get_ylim()) + 1e-5)))
# #         ax.yaxis.set_major_formatter(
# #             OOMFormatter(order=int(order_y), fformat=fformat)
# #         )
# #         ax.ticklabel_format(axis="y", style="sci", scilimits=scilimits)
# 
# #     return ax
# 
# 
# # #!/usr/bin/env python3
# 
# 
# # import matplotlib
# 
# 
# # class OOMFormatter(matplotlib.ticker.ScalarFormatter):
# #     # https://stackoverflow.com/questions/42656139/set-scientific-notation-with-fixed-exponent-and-significant-digits-for-multiple
# #     # def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
# #     def __init__(self, order=0, fformat="%1.0d", offset=True, mathText=True):
# #         self.oom = order
# #         self.fformat = fformat
# #         matplotlib.ticker.ScalarFormatter.__init__(
# #             self, useOffset=offset, useMathText=mathText
# #         )
# 
# #     def _set_order_of_magnitude(self):
# #         self.orderOfMagnitude = self.oom
# 
# #     def _set_format(self, vmin=None, vmax=None):
# #         self.format = self.fformat
# #         if self._useMathText:
# #             self.format = r"$\mathdefault{%s}$" % self.format
# 
# 
# # def sci_note(
# #     ax,
# #     order,
# #     fformat="%1.0d",
# #     x=False,
# #     y=False,
# #     scilimits=(-3, 3),
# # ):
# #     """
# #     Change the expression of the x- or y-axis to the scientific notation like *10^3
# #     , where 3 is the first argument, order.
# 
# #     Example:
# #         order = 4 # 10^4
# #         ax = sci_note(
# #                  ax,
# #                  order,
# #                  fformat="%1.0d",
# #                  x=True,
# #                  y=False,
# #                  scilimits=(-3, 3),
# #     """
# 
# #     if x == True:
# #         ax.xaxis.set_major_formatter(
# #             OOMFormatter(order=order, fformat=fformat)
# #         )
# #         ax.ticklabel_format(axis="x", style="sci", scilimits=scilimits)
# #     if y == True:
# #         ax.yaxis.set_major_formatter(
# #             OOMFormatter(order=order, fformat=fformat)
# #         )
# #         ax.ticklabel_format(axis="y", style="sci", scilimits=scilimits)
# 
# #     return ax

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs..plt.ax._sci_note import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        pass

    def test_edge_cases(self):
        # Edge case testing
        pass

    def test_error_handling(self):
        # Error handling testing
        pass
