# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_colors.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# 
# import numpy as np
# 
# from ._PARAMS import PARAMS
# from matplotlib import colors as _colors
# 
# 
# def to_rgb(c):
#     return PARAMS["RGB"][c]
# 
# 
# def to_rgba(c, alpha=0.5):
#     rgba = rgb2rgba(PARAMS["RGB"][c])
#     rgba[-1] = alpha
#     return rgba
# 
# 
# def to_hex(c):
#     return PARAMS["HEX"][c]
# 
# 
# def rgb2rgba(rgb, alpha=0, round=2):
#     rgb = np.array(rgb).astype(float)
#     rgb /= 255
#     return [*rgb.round(round), alpha]
# 
# 
# def rgba2rgb(rgba):
#     rgba = np.array(rgba).astype(float)
#     rgb = (rgba[:3] * 255).clip(0, 255)
#     return rgb.round(2).tolist()
# 
# 
# def update_alpha(rgba, alpha):
#     rgba_list = list(rgba)
#     rgba_list[-1] = alpha
#     return rgba_list
# 
# 
# def rgba2hex(rgba):
#     return "#{:02x}{:02x}{:02x}{:02x}".format(
#         int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3] * 255)
#     )
# 
# 
# def cycle_color(i_color):
#     COLORS_10_STR = list(PARAMS["RGB"].keys())
#     n_colors = len(COLORS_10_STR)
#     return COLORS_10_STR[i_color % n_colors]
# 
# 
# def gradiate_color(rgb_or_rgba, n=5):
#     has_alpha = len(rgb_or_rgba) == 4
#     alpha = None
# 
#     if has_alpha:
#         alpha = rgb_or_rgba[3]
#         rgba = rgb_or_rgba
#         rgb = rgba2rgb(rgba)
#     else:
#         rgb = rgb_or_rgba
# 
#     rgb_hsv = _colors.rgb_to_hsv([v for v in rgb])
#     # gradient_dict = {}
#     gradient = []
# 
#     for step in range(n):
#         color_hsv = [
#             rgb_hsv[0],
#             rgb_hsv[1],
#             rgb_hsv[2] * (1.0 - (step / n)),
#         ]
#         color_rgb = [int(v) for v in _colors.hsv_to_rgb(color_hsv)]
#         if has_alpha:
#             gradient.append(rgb2rgba(color_rgb, alpha=alpha))
#         else:
#             gradient.append(color_rgb)
# 
#     return gradient
# 
# 
# if __name__ == "__main__":
#     c = "blue"
#     print(to_rgb(c))
#     print(to_rgba(c))
#     print(to_hex(c))
#     print(cycle_color(1))

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

from mngs.plt._colors import *

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
