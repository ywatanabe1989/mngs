#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:35:17 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/test__imshow2d.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/test__imshow2d.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
from mngs.plt.ax._imshow2d import imshow2d


class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.data_2d = np.random.rand(10, 20)  # 10 rows, 20 columns

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)

    def test_basic_functionality(self):
        # Test basic display with default parameters
        ax = imshow2d(self.ax, self.data_2d)

        # Check that there's an image
        assert len(ax.images) == 1

        # Check that y-axis is inverted
        assert ax.get_ylim()[0] < ax.get_ylim()[1]

        # Check that colorbar is shown
        assert len(self.fig.axes) > 1

    def test_without_colorbar(self):
        # Test without colorbar
        ax = imshow2d(self.ax, self.data_2d, cbar=False)

        # Check that no colorbar is added
        assert len(self.fig.axes) == 1

    def test_with_colorbar_label(self):
        # Test with colorbar label
        cbar_label = "Test Label"
        ax = imshow2d(self.ax, self.data_2d, cbar_label=cbar_label)

        # Check that colorbar label is correctly set
        colorbar_axes = None
        for axes in self.fig.axes:
            if axes != self.ax:
                colorbar_axes = axes
                break

        assert colorbar_axes is not None
        # Note: Colorbar label is attached to the colorbar axes
        assert colorbar_axes.get_ylabel() == cbar_label

    def test_with_custom_cmap(self):
        # Test with custom colormap
        custom_cmap = "hot"
        ax = imshow2d(self.ax, self.data_2d, cmap=custom_cmap)

        # Check that the colormap is correctly set
        assert ax.images[0].get_cmap().name == custom_cmap

    def test_with_custom_vmin_vmax(self):
        # Test with custom vmin and vmax
        vmin, vmax = 0.2, 0.8
        ax = imshow2d(self.ax, self.data_2d, vmin=vmin, vmax=vmax)

        # Check that vmin and vmax are correctly set
        assert ax.images[0].norm.vmin == vmin
        assert ax.images[0].norm.vmax == vmax

    def test_with_custom_aspect(self):
        # Test with custom aspect
        aspect = "equal"
        ax = imshow2d(self.ax, self.data_2d, aspect=aspect)

        # Check that aspect is correctly set
        assert ax.images[0].get_aspect() == aspect

    def test_error_handling(self):
        # Test with invalid input
        with pytest.raises(AssertionError):
            # Should fail with 1D array
            imshow2d(self.ax, np.random.rand(10))

        with pytest.raises(AssertionError):
            # Should fail with 3D array
            imshow2d(self.ax, np.random.rand(10, 20, 3))


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_imshow2d.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-09-28 10:19:09 (ywatanabe)"
# # Author: Yusuke Watanabe (ywata1989@gmail.com)
#
# """This script does XYZ."""
#
# import sys
#
# import matplotlib.pyplot as plt
# import mngs
# import numpy as np
#
#
# # Functions
# def imshow2d(
#     ax,
#     arr_2d,
#     cbar=True,
#     cbar_label=None,
#     cbar_shrink=1.0,
#     cbar_fraction=0.046,
#     cbar_pad=0.04,
#     cmap="viridis",
#     aspect="auto",
#     vmin=None,
#     vmax=None,
#     **kwargs,
# ):
#     """
#     Imshows an two-dimensional array with theese two conditions:
#     1) The first dimension represents the x dim, from left to right.
#     2) The second dimension represents the y dim, from bottom to top
#     """
#
#     assert arr_2d.ndim == 2
#
#     # Transposes arr_2d for correct orientation
#     arr_2d = arr_2d.T
#
#     # Cals the original ax.imshow() method on the transposed array
#     im = ax.imshow(
#         arr_2d, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, **kwargs
#     )
#
#     # Color bar
#     if cbar:
#         fig = ax.get_figure()
#         _cbar = fig.colorbar(
#             im, ax=ax, shrink=cbar_shrink, fraction=cbar_fraction, pad=cbar_pad
#         )
#         if cbar_label:
#             _cbar.set_label(cbar_label)
#
#     # Invert y-axis to match typical image orientation
#     ax.invert_yaxis()
#
#     return ax
#
#
# if __name__ == "__main__":
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
#
#     # Example usage
#     fig, ax = plt.subplots()
#     data = np.random.rand(10, 20)  # Random data
#     imshow2d(ax, data)
#     plt.show()
#
#     # Close
#     mngs.gen.close(CONFIG)
#
# # EOF
#
# """
# /ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/plt/ax/_imshow2d.py
# """

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_imshow2d.py
# --------------------------------------------------------------------------------

# EOF