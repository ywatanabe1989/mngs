#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:31:44 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/test__fillv.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/test__fillv.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mngs.plt.ax._fillv import fillv

matplotlib.use("Agg")


class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)))

        # Setup multiple axes
        self.fig_multi = plt.figure()
        self.axes = np.array(
            [
                self.fig_multi.add_subplot(2, 2, 1),
                self.fig_multi.add_subplot(2, 2, 2),
                self.fig_multi.add_subplot(2, 2, 3),
                self.fig_multi.add_subplot(2, 2, 4),
            ]
        )

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)
        plt.close(self.fig_multi)

    def test_basic_functionality(self):
        # Test filling single region on a single axis
        starts = [2]
        ends = [4]
        ax = fillv(self.ax, starts, ends)

        # Should create one span patch
        assert (
            len(
                [
                    p
                    for p in self.ax.patches
                    if isinstance(p, matplotlib.patches.Polygon)
                ]
            )
            == 1
        )

        # Check that ax is returned
        assert ax is self.ax

    def test_multiple_regions(self):
        # Test filling multiple regions on a single axis
        starts = [1, 3, 5, 7, 9]
        ends = [2, 4, 6, 8, 10]
        ax = fillv(self.ax, starts, ends)

        # Should create five span patches
        assert (
            len(
                [
                    p
                    for p in self.ax.patches
                    if isinstance(p, matplotlib.patches.Polygon)
                ]
            )
            == 5
        )

    def test_multiple_axes(self):
        # Test filling regions on multiple axes
        starts = [2, 4]
        ends = [3, 5]
        axes = fillv(self.axes, starts, ends)

        # Should create patches on all axes
        for ax in self.axes:
            assert (
                len(
                    [
                        p
                        for p in ax.patches
                        if isinstance(p, matplotlib.patches.Polygon)
                    ]
                )
                == 2
            )

        # Should return the axes array
        assert np.array_equal(axes, self.axes)

    def test_custom_color_alpha(self):
        # Test with custom color and alpha
        starts = [2]
        ends = [4]
        color = "blue"
        alpha = 0.8
        ax = fillv(self.ax, starts, ends, color=color, alpha=alpha)

        # Get the span patch
        patches = [
            p
            for p in self.ax.patches
            if isinstance(p, matplotlib.patches.Polygon)
        ]
        assert len(patches) == 1

        # Check patch properties
        patch = patches[0]
        assert patch.get_fc()[0:3] == matplotlib.colors.to_rgb(color)
        assert patch.get_alpha() == alpha

    def test_edge_cases(self):
        # Test with empty lists
        ax = fillv(self.ax, [], [])

        # Should not create any patches
        assert (
            len(
                [
                    p
                    for p in self.ax.patches
                    if isinstance(p, matplotlib.patches.Polygon)
                ]
            )
            == 0
        )

        # Test with mismatched lists
        with pytest.raises(Exception):
            fillv(self.ax, [1, 2], [3])  # Different lengths


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_fillv.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-07-05 06:01:32 (ywatanabe)"
# # ./src/mngs/plt/ax/_fill_between_v.py
#
#
# """
# This script does XYZ.
# """
#
#
# """
# Imports
# """
#
# import numpy as np
#
#
# def fillv(axes, starts, ends, color="red", alpha=0.2):
#     """
#     Fill between specified start and end intervals on an axis or array of axes.
#
#     Parameters
#     ----------
#     axes : matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes
#         The axis object(s) to fill intervals on.
#     starts : array-like
#         Array-like of start positions.
#     ends : array-like
#         Array-like of end positions.
#     color : str, optional
#         The color to use for the filled regions. Default is "red".
#     alpha : float, optional
#         The alpha blending value, between 0 (transparent) and 1 (opaque). Default is 0.2.
#
#     Returns
#     -------
#     list
#         List of axes with filled intervals.
#     """
#
#     is_axes = isinstance(axes, np.ndarray)
#
#     axes = axes if isinstance(axes, np.ndarray) else [axes]
#
#     for ax in axes:
#         for start, end in zip(starts, ends):
#             ax.axvspan(start, end, color=color, alpha=alpha)
#
#     if not is_axes:
#         return axes[0]
#     else:
#         return axes
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_fillv.py
# --------------------------------------------------------------------------------

# EOF