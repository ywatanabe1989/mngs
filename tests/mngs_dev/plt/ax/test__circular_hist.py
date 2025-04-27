#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 19:24:50 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/test__circular_hist.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/test__circular_hist.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import sys

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/ax/_circular_hist.py
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


# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
matplotlib.use("Agg")

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))
from mngs.plt.ax._circular_hist import circular_hist


class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures - polar axes required for circular histogram
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="polar")

        # Create sample radians data (0 to 2pi)
        self.rads = np.random.uniform(0, 2 * np.pi, 1000)

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)

    def test_basic_functionality(self):
        # Test with default parameters
        n, bins, patches = circular_hist(self.ax, self.rads)

        # Check return values
        assert isinstance(n, np.ndarray)
        assert isinstance(bins, np.ndarray)
        assert len(n) == len(bins) - 1
        assert len(n) == 16  # Default bin count

        # Check that patches were added to the plot
        assert len(patches) == 16

    def test_with_custom_bins(self):
        # Test with custom number of bins
        bin_count = 24
        n, bins, patches = circular_hist(self.ax, self.rads, bins=bin_count)

        # Check correct number of bins
        assert len(n) == bin_count
        assert len(patches) == bin_count

    def test_with_no_gaps(self):
        # Test with gaps=False
        n, bins, patches = circular_hist(self.ax, self.rads, gaps=False)

        # Check that bins span the entire circle
        assert np.isclose(bins[0], -np.pi)
        assert np.isclose(bins[-1], np.pi)

    def test_with_custom_color(self):
        # Test with custom color
        color = "red"
        n, bins, patches = circular_hist(self.ax, self.rads, color=color)

        # Check that patches have the correct color
        for patch in patches:
            assert patch.get_edgecolor()[0:3] == matplotlib.colors.to_rgb(
                color
            )

    def test_with_non_density(self):
        # Test with density=False
        n, bins, patches = circular_hist(self.ax, self.rads, density=False)

        # Check that y-ticks are visible
        assert len(self.ax.get_yticks()) > 0

    def test_with_offset(self):
        # Test with custom offset
        offset = np.pi / 4  # 45 degrees
        n, bins, patches = circular_hist(self.ax, self.rads, offset=offset)

        # Check that theta offset was set correctly
        assert self.ax.get_theta_offset() == offset

    def test_with_range_bias(self):
        # Test with range_bias
        range_bias = 0.5
        n, bins, patches = circular_hist(
            self.ax, self.rads, range_bias=range_bias
        )

        # Check that histogram is biased as expected
        assert np.isclose(bins[0], -np.pi + range_bias, atol=1e-5)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF