#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 19:24:56 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/test__extend.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/test__extend.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/ax/_extend.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
#
#
# def extend(ax, x_ratio=1.0, y_ratio=1.0):
#     ## Original coordinates
#     bbox = ax.get_position()
#     left_orig = bbox.x0
#     bottom_orig = bbox.y0
#     width_orig = bbox.x1 - bbox.x0
#     height_orig = bbox.y1 - bbox.y0
#     g_orig = (left_orig + width_orig / 2.0, bottom_orig + height_orig / 2.0)
#
#     ## Target coordinates
#     g_tgt = g_orig
#     width_tgt = width_orig * x_ratio
#     height_tgt = height_orig * y_ratio
#     left_tgt = g_tgt[0] - width_tgt / 2
#     bottom_tgt = g_tgt[1] - height_tgt / 2
#
#     ax.set_position(
#         [
#             left_tgt,
#             bottom_tgt,
#             width_tgt,
#             height_tgt,
#         ]
#     )
#     return ax
#
#
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     fig, axes = plt.subplots(1, 2)
#     ax = axes[1]
#     ax = extend(ax, 0.75, 1.1)
#     fig.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))
from mngs.plt.ax._extend import extend


class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        self.fig = plt.figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)

    def test_basic_functionality(self):
        # Get original position
        original_bbox = self.ax.get_position()
        original_width = original_bbox.width
        original_height = original_bbox.height
        original_center_x = original_bbox.x0 + original_width / 2
        original_center_y = original_bbox.y0 + original_height / 2

        # Extend width by 50% and keep height same
        extended_ax = extend(self.ax, x_ratio=1.5, y_ratio=1.0)
        new_bbox = extended_ax.get_position()

        # Check that center point remains the same
        new_center_x = new_bbox.x0 + new_bbox.width / 2
        new_center_y = new_bbox.y0 + new_bbox.height / 2
        assert np.isclose(new_center_x, original_center_x)
        assert np.isclose(new_center_y, original_center_y)

        # Check that width and height were scaled correctly
        assert np.isclose(new_bbox.width, original_width * 1.5)
        assert np.isclose(new_bbox.height, original_height * 1.0)

    def test_shrink(self):
        # Test shrinking the axes
        original_bbox = self.ax.get_position()
        original_width = original_bbox.width
        original_height = original_bbox.height

        # Shrink width and height by 50%
        extended_ax = extend(self.ax, x_ratio=0.5, y_ratio=0.5)
        new_bbox = extended_ax.get_position()

        # Check that width and height were scaled correctly
        assert np.isclose(new_bbox.width, original_width * 0.5)
        assert np.isclose(new_bbox.height, original_height * 0.5)

    def test_asymmetric_scaling(self):
        # Test different scaling for width and height
        original_bbox = self.ax.get_position()
        original_width = original_bbox.width
        original_height = original_bbox.height

        # Extend width but shrink height
        extended_ax = extend(self.ax, x_ratio=2.0, y_ratio=0.75)
        new_bbox = extended_ax.get_position()

        # Check that width and height were scaled correctly
        assert np.isclose(new_bbox.width, original_width * 2.0)
        assert np.isclose(new_bbox.height, original_height * 0.75)

    def test_edge_cases(self):
        # Test with zero scaling (should be invalid but testing edge case)
        with pytest.raises(Exception):
            extend(self.ax, x_ratio=0, y_ratio=0)

        # Test with default values (should keep size the same)
        original_bbox = self.ax.get_position()
        extended_ax = extend(self.ax)
        new_bbox = extended_ax.get_position()

        assert np.isclose(new_bbox.width, original_bbox.width)
        assert np.isclose(new_bbox.height, original_bbox.height)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])

# EOF