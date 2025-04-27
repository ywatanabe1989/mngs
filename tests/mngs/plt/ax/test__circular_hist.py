#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 13:29:43 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/test__circular_hist.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/test__circular_hist.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_circular_hist.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-02-03 13:10:50 (ywatanabe)"
# import numpy as np
#
#
# def circular_hist(
#     ax,
#     rads,
#     bins=16,
#     density=True,
#     offset=0,
#     gaps=True,
#     color=None,
#     range_bias=0,
# ):
#     """
#     Example:
#         fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
#         ax = mngs.plt.circular_hist(ax, rads)
#     Produce a circular histogram of angles on ax.
#
#     Parameters
#     ----------
#     ax : matplotlib.axes._subplots.PolarAxesSubplot
#         axis instance created with subplot_kw=dict(projection='polar').
#
#     rads : array
#         Angles to plot, expected in units of radians.
#
#     bins : int, optional
#         Defines the number of equal-width bins in the range. The default is 16.
#
#     density : bool, optional
#         If True plot frequency proportional to area. If False plot frequency
#         proportional to radius. The default is True.
#
#     offset : float, optional
#         Sets the offset for the location of the 0 direction in units of
#         radians. The default is 0.
#
#     gaps : bool, optional
#         Whether to allow gaps between bins. When gaps = False the bins are
#         forced to partition the entire [-pi, pi] range. The default is True.
#
#     Returns
#     -------
#     n : array or list of arrays
#         The number of values in each bin.
#
#     bins : array
#         The edges of the bins.
#
#     patches : `.BarContainer` or list of a single `.Polygon`
#         Container of individual artists used to create the histogram
#         or list of such containers if there are multiple input datasets.
#     """
#     # Wrap angles to [-pi, pi)
#     rads = (rads + np.pi) % (2 * np.pi) - np.pi
#
#     # Force bins to partition entire circle
#     if not gaps:
#         bins = np.linspace(-np.pi, np.pi, num=bins + 1)
#
#     # Bin data and record counts
#     n, bins = np.histogram(
#         rads, bins=bins, range=(-np.pi + range_bias, np.pi + range_bias)
#     )
#
#     # Compute width of each bin
#     widths = np.diff(bins)
#
#     # By default plot frequency proportional to area
#     if density:
#         # Area to assign each bin
#         area = n / rads.size
#         # Calculate corresponding bin radius
#         radius = (area / np.pi) ** 0.5
#     # Otherwise plot frequency proportional to radius
#     else:
#         radius = n
#
#     # fixme
#     # med_val = np.pi/2#
#     # med_val = np.nanmedian(rads)
#     mean_val = np.nanmean(rads)
#     std_val = np.nanstd(rads)
#     ax.axvline(mean_val, color=color)
#     ax.text(mean_val, 1, std_val)
#
#     # Plot data on ax
#     patches = ax.bar(
#         bins[:-1],
#         radius,
#         zorder=1,
#         align="edge",
#         width=widths,
#         # edgecolor="C0",
#         edgecolor=color,
#         alpha=0.9,
#         fill=False,
#         linewidth=1,
#     )
#
#     # Set the direction of the zero angle
#     ax.set_theta_offset(offset)
#
#     # Remove ylabels for area plots (they are mostly obstructive)
#     if density:
#         ax.set_yticks([])
#
#     return n, bins, patches

import sys

#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.plt.ax._circular_hist import circular_hist


def test_circular_hist_counts_and_bins():
    #  uniform angles
    angles = np.linspace(-np.pi, np.pi, 64)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    counts, bins, patches = circular_hist(
        ax, angles, bins=8, density=False, color="k"
    )
    assert isinstance(counts, np.ndarray)
    assert counts.sum() == angles.size
    assert len(bins) == 9


def test_circular_hist_density():
    angles = np.random.rand(100) * 2 * np.pi - np.pi
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    counts, bins, patches = circular_hist(ax, angles, bins=16, density=True)
    assert np.all(counts >= 0)

# EOF