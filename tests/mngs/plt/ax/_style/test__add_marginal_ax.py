# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_style/_add_marginal_ax.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-30 20:18:52 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_add_marginal_ax.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/ax/_add_marginal_ax.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import matplotlib
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# 
# 
# def add_marginal_ax(axis, place, size=0.2, pad=0.1):
#     assert isinstance(
#         axis, matplotlib.axes._axes.Axes
#     ), "First argument must be a matplotlib axis"
# 
#     divider = make_axes_locatable(axis)
# 
#     size_perc_str = f"{size*100}%"
#     if place in ["left", "right"]:
#         size = 1.0 / size
# 
#     axis_marginal = divider.append_axes(place, size=size_perc_str, pad=pad)
#     axis_marginal.set_box_aspect(size)
# 
#     return axis_marginal
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_style/_add_marginal_ax.py
# --------------------------------------------------------------------------------
