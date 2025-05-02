# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_style/_extend.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-02 09:00:51 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_style/_extend.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/ax/_style/_extend.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import matplotlib
# 
# 
# def extend(axis, x_ratio=1.0, y_ratio=1.0):
#     """
#     Extend or shrink a matplotlib axis while maintaining its center position.
# 
#     Args:
#         axis (matplotlib.axes._axes.Axes): The matplotlib axis to be modified.
#         x_ratio (float, optional): The ratio to scale the width. Default is 1.0.
#         y_ratio (float, optional): The ratio to scale the height. Default is 1.0.
# 
#     Returns:
#         matplotlib.axes._axes.Axes: The modified matplotlib axis.
# 
#     Raises:
#         AssertionError: If the first argument is not a matplotlib axis.
#     """
# 
#     assert isinstance(
#         axis, matplotlib.axes._axes.Axes
#     ), "First argument must be a matplotlib axis"
# 
#     assert x_ratio != 0, "x_ratio must not be 0."
#     assert y_ratio != 0, "y_ratio must not be 0."
# 
#     ## Original coordinates
#     bbox = axis.get_position()
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
#     # Extend the axis
#     axis.set_position(
#         [
#             left_tgt,
#             bottom_tgt,
#             width_tgt,
#             height_tgt,
#         ]
#     )
#     return axis
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_style/_extend.py
# --------------------------------------------------------------------------------
