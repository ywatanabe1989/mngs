#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 19:25:29 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/test__set_n_ticks.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/test__set_n_ticks.py"
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

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/ax/_set_n_ticks.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python
#
#
# import matplotlib
#
#
# def set_n_ticks(
#     ax,
#     n_xticks=4,
#     n_yticks=4,
# ):
#     """
#     Example:
#         ax = set_n_ticks(ax)
#     """
#
#     if n_xticks is not None:
#         ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(n_xticks))
#
#     if n_yticks is not None:
#         ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(n_yticks))
#
#     # Force the figure to redraw to reflect changes
#     ax.figure.canvas.draw()
#
#     return ax


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
from mngs.plt.ax._set_n_ticks import set_n_ticks


class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        # Create a basic plot with many potential tick locations
        xx = np.linspace(0, 100, 1000)
        yy = np.sin(xx * 0.1)
        self.ax.plot(xx, yy)

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)

    def test_basic_functionality(self):
        # Test with default parameters (4 ticks)
        ax = set_n_ticks(self.ax)

        # Force draw to ensure ticks are updated
        self.fig.canvas.draw()

        # Only count visible ticks
        visible_xticks = len(
            [t for t in ax.xaxis.get_major_ticks() if t.get_visible()]
        )
        visible_yticks = len(
            [t for t in ax.yaxis.get_major_ticks() if t.get_visible()]
        )

        # Should be approximately 4 ticks
        assert 3 <= visible_xticks <= 5
        assert 3 <= visible_yticks <= 5

    def test_custom_tick_counts(self):
        # Test with custom number of ticks
        ax = set_n_ticks(self.ax, n_xticks=6, n_yticks=3)

        # Force draw to ensure ticks are updated
        self.fig.canvas.draw()

        # Only count visible ticks
        visible_xticks = len(
            [t for t in ax.xaxis.get_major_ticks() if t.get_visible()]
        )
        visible_yticks = len(
            [t for t in ax.yaxis.get_major_ticks() if t.get_visible()]
        )

        # Should be approximately the requested number of ticks
        assert 5 <= visible_xticks <= 7
        assert 2 <= visible_yticks <= 4

    def test_x_ticks_only(self):
        # Test setting only x ticks
        ax = set_n_ticks(self.ax, n_xticks=7, n_yticks=None)

        # Force draw to ensure ticks are updated
        self.fig.canvas.draw()

        # Check x ticks change but y ticks remain default
        visible_xticks = len(
            [t for t in ax.xaxis.get_major_ticks() if t.get_visible()]
        )
        assert 6 <= visible_xticks <= 8

    def test_y_ticks_only(self):
        # Test setting only y ticks
        ax = set_n_ticks(self.ax, n_xticks=None, n_yticks=7)

        # Force draw to ensure ticks are updated
        self.fig.canvas.draw()

        # Check y ticks change but x ticks remain default
        visible_yticks = len(
            [t for t in ax.yaxis.get_major_ticks() if t.get_visible()]
        )
        assert 6 <= visible_yticks <= 8

    def test_error_handling(self):
        # Test with invalid input types
        with pytest.raises(Exception):
            set_n_ticks(self.ax, n_xticks="invalid")


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF