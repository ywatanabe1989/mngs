# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_extend.py
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
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.plt.ax._extend import *

class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        raise NotImplementedError("Test not yet implemented")

    def test_edge_cases(self):
        # Edge case testing
        raise NotImplementedError("Test not yet implemented")

    def test_error_handling(self):
        # Error handling testing
        raise NotImplementedError("Test not yet implemented")

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
