#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 16:38:32 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__colors.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__colors.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/_colors.py
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

import sys

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.plt._colors import (
    cycle_color,
    gradiate_color,
    rgb2rgba,
    rgba2hex,
    rgba2rgb,
    to_hex,
    to_rgb,
    update_alpha,
)


def test_to_rgb_and_to_hex():
    assert to_rgb("red") == [255, 70, 50]
    assert to_hex("blue").startswith("#")


def test_rgb2rgba_and_rgba2rgb_and_update_alpha():
    rgba = rgb2rgba([10, 20, 30], alpha=0.5, round=2)
    assert isinstance(rgba, list)
    assert rgba[-1] == 0.5
    rgb = rgba2rgb([0.2, 0.4, 0.6, 0.8])
    assert isinstance(rgb, list) and len(rgb) == 3
    updated = update_alpha([1, 2, 3, 0.1], 0.9)
    assert updated[-1] == 0.9


def test_rgba2hex_and_cycle_and_gradiate():
    hexv = rgba2hex([255, 255, 255, 1])
    assert hexv.lower() == "#ffffffff"
    c = cycle_color(0)
    assert isinstance(c, str)
    grad = gradiate_color([100, 150, 200], n=3)
    assert isinstance(grad, list) and len(grad) == 3

# EOF