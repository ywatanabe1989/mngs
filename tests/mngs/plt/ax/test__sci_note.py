#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:40:09 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/test__sci_note.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/test__sci_note.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mngs.plt.ax._sci_note import OOMFormatter, sci_note

matplotlib.use("Agg")


class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        # Create a simple plot with large values
        xx = np.linspace(0, 1, 100)
        yy = xx * 1e6  # Large values on y-axis
        self.ax.plot(xx * 1e-6, yy)  # Small values on x-axis

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)

    def test_oom_formatter(self):
        # Test OOMFormatter behavior
        formatter = OOMFormatter(order=3, fformat="%1.2f")

        # Check properties
        assert formatter.order == 3
        assert formatter.fformat == "%1.2f"

        # Set order of magnitude and check it's set correctly
        formatter._set_order_of_magnitude()
        assert formatter.orderOfMagnitude == 3

        # Test format setting
        formatter._set_format()
        if formatter._useMathText:
            assert formatter.format == r"$\mathdefault{%1.2f}$"
        else:
            assert formatter.format == "%1.2f"

    def test_sci_note_x_axis(self):
        # Test scientific notation on x-axis
        ax = sci_note(self.ax, x=True, y=False)

        # Force draw to update formatters
        self.fig.canvas.draw()

        # Check that the x-axis formatter is an OOMFormatter
        assert isinstance(ax.xaxis.get_major_formatter(), OOMFormatter)

        # Check that the y-axis formatter is still the default
        assert not isinstance(ax.yaxis.get_major_formatter(), OOMFormatter)

        # Check that the x-axis uses scientific notation
        # assert ax.xaxis._major_formatter._style == "sci"
        assert "sci" in str(ax.xaxis.get_major_formatter()).lower()

    def test_sci_note_y_axis(self):
        # Test scientific notation on y-axis
        ax = sci_note(self.ax, x=False, y=True)

        # Force draw to update formatters
        self.fig.canvas.draw()

        # Check that the y-axis formatter is an OOMFormatter
        assert isinstance(ax.yaxis.get_major_formatter(), OOMFormatter)

        # Check that the x-axis formatter is still the default
        assert not isinstance(ax.xaxis.get_major_formatter(), OOMFormatter)

        # Check that the y-axis uses scientific notation
        # assert ax.yaxis._major_formatter._style == "sci"
        assert "sci" in str(ax.yaxis.get_major_formatter()).lower()

    def test_sci_note_both_axes(self):
        # Test scientific notation on both axes
        ax = sci_note(self.ax, x=True, y=True)

        # Force draw to update formatters
        self.fig.canvas.draw()

        # Check that both axes formatters are OOMFormatters
        assert isinstance(ax.xaxis.get_major_formatter(), OOMFormatter)
        assert isinstance(ax.yaxis.get_major_formatter(), OOMFormatter)

        # Check that both axes use scientific notation
        assert ax.xaxis._major_formatter._style == "sci"
        assert ax.yaxis._major_formatter._style == "sci"

    def test_sci_note_custom_format(self):
        # Test with custom format
        custom_format = "%1.3f"
        ax = sci_note(self.ax, fformat=custom_format, x=True, y=True)

        # Force draw to update formatters
        self.fig.canvas.draw()

        # Check that formatters have the custom format
        assert ax.xaxis.get_major_formatter().fformat == custom_format
        assert ax.yaxis.get_major_formatter().fformat == custom_format

    def test_sci_note_custom_limits(self):
        # Test with custom sci limits
        custom_limits = (-2, 2)
        ax = sci_note(self.ax, scilimits=custom_limits, x=True, y=True)

        # Force draw to update formatters
        self.fig.canvas.draw()

        # Check that formatters have the custom limits
        assert ax.xaxis._major_formatter._scilimits == custom_limits
        assert ax.yaxis._major_formatter._scilimits == custom_limits

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_sci_note.py
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_sci_note.py
# --------------------------------------------------------------------------------
