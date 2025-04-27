#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 19:32:11 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/test__rectangle.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/test__rectangle.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
from pathlib import Path

import matplotlib

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/ax/_rectangle.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-15 18:55:49"
# # Author: Yusuke Watanabe (ywata1989@gmail.com)
#
# """
# This script does XYZ.
# """
#
# import sys
#
# import matplotlib.pyplot as plt
#
# # Imports
# import mngs
# from matplotlib.patches import Rectangle
#
#
# # Functions
# def rectangle(ax, xx, yy, ww, hh, **kwargs):
#     ax.add_patch(Rectangle((xx, yy), ww, hh, **kwargs))
#     return ax
#
#
# # (YOUR AWESOME CODE)
#
# if __name__ == "__main__":
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
#
#     # (YOUR AWESOME CODE)
#
#     # Close
#     mngs.gen.close(CONFIG)
#
# # EOF
#
# """
# /ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/plt/ax/_rectangle.py
# """

import sys

import matplotlib.pyplot as plt

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

matplotlib.use("Agg")
from matplotlib.patches import Rectangle
from mngs.plt.ax._rectangle import rectangle


class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)

    def test_basic_functionality(self):
        # Test creating a basic rectangle
        xx, yy = 1.0, 2.0  # Bottom-left corner
        ww, hh = 3.0, 4.0  # Width and height

        # Add rectangle to the plot
        ax = rectangle(self.ax, xx, yy, ww, hh)

        # Check that a patch was added to the axis
        assert len(ax.patches) == 1

        # Check that the patch is a Rectangle with correct properties
        patch = ax.patches[0]
        assert isinstance(patch, Rectangle)
        assert patch.get_x() == xx
        assert patch.get_y() == yy
        assert patch.get_width() == ww
        assert patch.get_height() == hh

    def test_with_styling(self):
        # Test creating a rectangle with custom styling
        xx, yy = 1.0, 2.0
        ww, hh = 3.0, 4.0
        color = "red"
        linewidth = 2.0
        alpha = 0.5

        # Add rectangle with styling
        ax = rectangle(
            self.ax,
            xx,
            yy,
            ww,
            hh,
            facecolor=color,
            linewidth=linewidth,
            alpha=alpha,
        )

        # Check that the patch has the correct styling
        patch = ax.patches[0]
        assert patch.get_facecolor()[0:3] == matplotlib.colors.to_rgb(color)
        assert patch.get_linewidth() == linewidth
        assert patch.get_alpha() == alpha

    def test_multiple_rectangles(self):
        # Test adding multiple rectangles
        # First rectangle
        ax = rectangle(self.ax, 0, 0, 1, 1, facecolor="red")

        # Second rectangle
        ax = rectangle(ax, 1, 1, 1, 1, facecolor="blue")

        # Third rectangle
        ax = rectangle(ax, 2, 2, 1, 1, facecolor="green")

        # Check that all three rectangles were added
        assert len(ax.patches) == 3

        # Check colors of rectangles
        assert ax.patches[0].get_facecolor()[0:3] == matplotlib.colors.to_rgb(
            "red"
        )
        assert ax.patches[1].get_facecolor()[0:3] == matplotlib.colors.to_rgb(
            "blue"
        )
        assert ax.patches[2].get_facecolor()[0:3] == matplotlib.colors.to_rgb(
            "green"
        )

    def test_edge_cases(self):
        # Test with zero width/height
        ax = rectangle(self.ax, 0, 0, 0, 0)
        patch = ax.patches[0]
        assert patch.get_width() == 0
        assert patch.get_height() == 0

        # Test with negative width/height (will be rendered correctly by matplotlib)
        ax = rectangle(self.ax, 5, 5, -1, -1)
        patch = ax.patches[1]
        assert patch.get_width() == -1
        assert patch.get_height() == -1


if __name__ == "__main__":
    import pytest

    pytest.main([os.path.abspath(__file__)])

# EOF