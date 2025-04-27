#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 19:25:12 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/test__hide_spines.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/test__hide_spines.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/ax/_hide_spines.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-26 20:03:45 (ywatanabe)"
#
#
# def hide_spines(
#     ax,
#     top=True,
#     bottom=True,
#     left=True,
#     right=True,
#     ticks=True,
#     labels=True,
# ):
#     """
#     Hides the specified spines of a matplotlib Axes object and optionally removes the ticks and labels.
#
#     This function is designed to work with matplotlib Axes objects. It allows for a cleaner, more minimalist
#     presentation of plots by hiding the spines (the lines denoting the boundaries of the plot area) and optionally
#     removing the ticks and labels from the axes.
#
#     Arguments:
#         ax (matplotlib.axes.Axes): The Axes object for which the spines will be hidden.
#         top (bool, optional): If True, hides the top spine. Defaults to True.
#         bottom (bool, optional): If True, hides the bottom spine. Defaults to True.
#         left (bool, optional): If True, hides the left spine. Defaults to True.
#         right (bool, optional): If True, hides the right spine. Defaults to True.
#         ticks (bool, optional): If True, removes the ticks from the hidden spines' axes. Defaults to True.
#         labels (bool, optional): If True, removes the labels from the hidden spines' axes. Defaults to True.
#
#     Returns:
#         matplotlib.axes.Axes: The modified Axes object with the specified spines hidden.
#
#     Example:
#         >>> fig, ax = plt.subplots()
#         >>> hide_spines(ax, top=False, labels=False)
#         >>> plt.show()
#     """
#
#     tgts = []
#     if top:
#         tgts.append("top")
#     if bottom:
#         tgts.append("bottom")
#     if left:
#         tgts.append("left")
#     if right:
#         tgts.append("right")
#
#     for tgt in tgts:
#         # Spines
#         ax.spines[tgt].set_visible(False)
#
#         # Ticks
#         if ticks:
#             if tgt == "bottom":
#                 ax.xaxis.set_ticks_position("none")
#             elif tgt == "left":
#                 ax.yaxis.set_ticks_position("none")
#
#         # Labels
#         if labels:
#             if tgt == "bottom":
#                 ax.set_xticklabels([])
#             elif tgt == "left":
#                 ax.set_yticklabels([])
#
#     return ax

import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))
from mngs.plt.ax._hide_spines import hide_spines


class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        self.fig = plt.figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        # Create a basic plot
        self.ax.plot([1, 2, 3], [1, 2, 3])

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)

    def test_hide_all_spines(self):
        # Test hiding all spines with default parameters
        ax = hide_spines(self.ax)

        # Check that all spines are hidden
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["bottom"].get_visible()
        assert not ax.spines["left"].get_visible()
        assert not ax.spines["right"].get_visible()

        # Check that ticks and labels are removed
        fig = ax.get_figure()
        fig.canvas.draw()
        assert len(ax.get_xticklabels()) > 0
        assert all(label.get_text() == "" for label in ax.get_xticklabels())
        assert all(label.get_text() == "" for label in ax.get_yticklabels())

    def test_hide_specific_spines(self):
        # Test hiding only specific spines
        ax = hide_spines(
            self.ax, top=True, bottom=False, left=True, right=False
        )

        # Check that only specified spines are hidden
        assert not ax.spines["top"].get_visible()
        assert ax.spines["bottom"].get_visible()
        assert not ax.spines["left"].get_visible()
        assert ax.spines["right"].get_visible()

    def test_keep_ticks_and_labels(self):
        # Test keeping ticks and labels
        ax = hide_spines(self.ax, ticks=False, labels=False)

        # Check that spines are hidden
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["bottom"].get_visible()
        assert not ax.spines["left"].get_visible()
        assert not ax.spines["right"].get_visible()

        # But ticks and labels should still be there
        fig = ax.get_figure()
        fig.canvas.draw()
        assert ax.xaxis.get_major_ticks() != []
        assert ax.yaxis.get_major_ticks() != []
        assert not all(
            label.get_text() == "" for label in ax.get_xticklabels()
        )
        assert not all(
            label.get_text() == "" for label in ax.get_yticklabels()
        )

    def test_hide_ticks_only(self):
        # Test hiding ticks but keeping labels
        ax = hide_spines(self.ax, ticks=True, labels=False)

        # Check that appropriate ticks are hidden
        assert ax.xaxis.get_ticks_position() == "none"
        assert ax.yaxis.get_ticks_position() == "none"

        # But labels should still be there
        fig = ax.get_figure()
        fig.canvas.draw()
        assert not all(
            label.get_text() == "" for label in ax.get_xticklabels()
        )
        assert not all(
            label.get_text() == "" for label in ax.get_yticklabels()
        )

    def test_hide_labels_only(self):
        # Test hiding labels but keeping ticks
        ax = hide_spines(self.ax, ticks=False, labels=True)

        # Check that labels are hidden
        fig = ax.get_figure()
        fig.canvas.draw()
        assert all(label.get_text() == "" for label in ax.get_xticklabels())
        assert all(label.get_text() == "" for label in ax.get_yticklabels())

        # But ticks should still be visible
        assert ax.xaxis.get_ticks_position() != "none"
        assert ax.yaxis.get_ticks_position() != "none"


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF