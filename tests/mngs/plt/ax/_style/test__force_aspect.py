# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_style/_force_aspect.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-02 09:00:52 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_style/_force_aspect.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/ax/_style/_force_aspect.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import matplotlib
# 
# 
# def force_aspect(axis, aspect=1):
#     assert isinstance(
#         axis, matplotlib.axes._axes.Axes
#     ), "First argument must be a matplotlib axes"
# 
#     im = axis.get_images()
# 
#     extent = im[0].get_extent()
# 
#     axis.set_aspect(
#         abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect
#     )
#     return axis
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_style/_force_aspect.py
# --------------------------------------------------------------------------------
