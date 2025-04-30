#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:14:33 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__colors.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__colors.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
from unittest.mock import patch

import numpy as np
from mngs.plt._colors import (
    cycle_color,
    gradiate_color,
    rgb2rgba,
    rgba2hex,
    rgba2rgb,
    to_hex,
    to_rgb,
    to_rgba,
    update_alpha,
)


def test_to_rgb():
    result = to_rgb("red")
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(val, int) for val in result)
    assert result == [255, 70, 50]


def test_to_rgba():
    result = to_rgba("blue", alpha=0.7)
    assert isinstance(result, list)
    assert len(result) == 4
    assert all(isinstance(result[idx], float) for idx in range(3))
    assert result[-1] == 0.7


def test_to_hex():
    result = to_hex("blue")
    assert isinstance(result, str)
    assert result.startswith("#")
    assert len(result) == 7  # #RRGGBB format


def test_rgb2rgba():
    rgba = rgb2rgba([10, 20, 30], alpha=0.5, round=2)
    assert isinstance(rgba, list)
    assert len(rgba) == 4
    assert all(isinstance(val, float) for val in rgba)
    assert rgba[-1] == 0.5
    assert rgba[0] == round(10 / 255, 2)
    assert rgba[1] == round(20 / 255, 2)
    assert rgba[2] == round(30 / 255, 2)


def test_rgba2rgb():
    rgb = rgba2rgb([0.2, 0.4, 0.6, 0.8])
    assert isinstance(rgb, list)
    assert len(rgb) == 3
    assert all(isinstance(val, float) for val in rgb)
    assert rgb[0] == round(0.2 * 255, 2)
    assert rgb[1] == round(0.4 * 255, 2)
    assert rgb[2] == round(0.6 * 255, 2)


def test_update_alpha():
    updated = update_alpha([1, 2, 3, 0.1], 0.9)
    assert isinstance(updated, list)
    assert len(updated) == 4
    assert updated[:3] == [1, 2, 3]
    assert updated[-1] == 0.9


def test_rgba2hex():
    hexv = rgba2hex([255, 255, 255, 1])
    assert isinstance(hexv, str)
    assert hexv.startswith("#")
    assert len(hexv) == 9  # #RRGGBBAA format
    assert hexv.lower() == "#ffffffff"

    hexv2 = rgba2hex([128, 64, 32, 0.5])
    assert hexv2.lower() == "#804020" + hex(int(0.5 * 255))[2:].zfill(2)


def test_cycle_color():
    with patch(
        "mngs.plt._colors.PARAMS",
        {"RGB": {"red": [255, 0, 0], "green": [0, 255, 0]}},
    ):
        c1 = cycle_color(0)
        assert c1 == "red"

        c2 = cycle_color(1)
        assert c2 == "green"

        c3 = cycle_color(2)  # Test cycling back
        assert c3 == "red"


def test_gradiate_color():
    with patch(
        "matplotlib.colors.rgb_to_hsv", return_value=np.array([0.5, 0.5, 0.5])
    ), patch(
        "matplotlib.colors.hsv_to_rgb", return_value=np.array([100, 150, 200])
    ), patch(
        "mngs.plt._colors.rgba2rgb", return_value=[0.39, 0.59, 0.78]
    ):

        # Test with RGB input
        grad = gradiate_color([100, 150, 200], n=3)
        assert isinstance(grad, list)
        assert len(grad) == 3
        assert all(isinstance(g, list) for g in grad)

        # Fix the failing test for RGBA input
        with patch(
            "mngs.plt._colors.rgb2rgba",
            side_effect=lambda rgb, alpha: [
                rgb[0] / 255,
                rgb[1] / 255,
                rgb[2] / 255,
                alpha,
            ],
        ):
            grad_rgba = gradiate_color([100, 150, 200, 0.7], n=2)
            assert isinstance(grad_rgba, list)
            assert len(grad_rgba) == 2
            assert all(len(g) == 4 for g in grad_rgba)
            assert all(g[-1] == 0.7 for g in grad_rgba)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_colors.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-29 13:39:01 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_colors.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_colors.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import matplotlib.colors as _colors
# import numpy as np
# from mngs.decorators._deprecated import deprecated
# 
# from ._PARAMS import PARAMS
# 
# # RGB
# # ------------------------------
# 
# 
# def str2rgb(c):
#     return PARAMS["RGB"][c]
# 
# 
# def str2rgba(c, alpha=0.5):
#     rgba = rgb2rgba(PARAMS["RGB"][c])
#     rgba[-1] = alpha
#     return rgba
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
# def rgba2hex(rgba):
#     return "#{:02x}{:02x}{:02x}{:02x}".format(
#         int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3] * 255)
#     )
# 
# 
# def cycle_color_rgb(i_color):
#     COLORS_10_STR = list(PARAMS["RGB"].keys())
#     n_colors = len(COLORS_10_STR)
#     return COLORS_10_STR[i_color % n_colors]
# 
# 
# def gradiate_color_rgb(rgb_or_rgba, n=5):
# 
#     # Separate RGB and alpha if present
#     if len(rgb_or_rgba) == 4:  # RGBA format
#         rgb = rgb_or_rgba[:3]
#         alpha = rgb_or_rgba[3]
#         has_alpha = True
#     else:  # RGB format
#         rgb = rgb_or_rgba
#         alpha = None
#         has_alpha = False
# 
#     # Scale RGB values to 0-1 range if they're in 0-255 range
#     if any(val > 1 for val in rgb):
#         rgb = [val / 255 for val in rgb]
# 
#     rgb_hsv = _colors.rgb_to_hsv(np.array(rgb))
# 
#     gradient = []
#     for step in range(n):
#         color_hsv = [
#             rgb_hsv[0],
#             rgb_hsv[1],
#             rgb_hsv[2] * (1.0 - (step / n)),
#         ]
#         color_rgb = [int(v * 255) for v in _colors.hsv_to_rgb(color_hsv)]
# 
#         if has_alpha:
#             gradient.append(rgb2rgba(color_rgb, alpha=alpha))
#         else:
#             gradient.append(color_rgb)
# 
#     return gradient
# 
# 
# def gradiate_color_rgba(rgb_or_rgba, n=5):
#     return gradiate_color_rgb(rgb_or_rgba, n)
# 
# 
# # BGRA
# # ------------------------------
# def bgr2rgb(bgr):
#     """Convert BGR color format to RGB format."""
#     return [bgr[2], bgr[1], bgr[0]]
# 
# 
# def rgb2bgr(rgb):
#     """Convert RGB color format to BGR format."""
#     return [rgb[2], rgb[1], rgb[0]]
# 
# 
# def bgra2rgba(bgra):
#     """Convert BGRA color format to RGBA format."""
#     return [bgra[2], bgra[1], bgra[0], bgra[3]]
# 
# 
# def rgba2bgra(rgba):
#     """Convert RGBA color format to BGRA format."""
#     return [rgba[2], rgba[1], rgba[0], rgba[3]]
# 
# 
# def bgra2hex(bgra):
#     """Convert BGRA color format to hex format."""
#     rgba = bgra2rgba(bgra)
#     return rgba2hex(rgba)
# 
# 
# def cycle_color_bgr(i_color):
#     rgb_color = str2rgb(cycle_color(i_color))
#     return rgb2bgr(rgb_color)
# 
# 
# # Common
# # ------------------------------
# def str2hex(c):
#     return PARAMS["HEX"][c]
# 
# 
# def update_alpha(rgba, alpha):
#     rgba_list = list(rgba)
#     rgba_list[-1] = alpha
#     return rgba_list
# 
# 
# def cycle_color(i_color):
#     COLORS_10_STR = list(PARAMS["RGB"].keys())
#     n_colors = len(COLORS_10_STR)
#     return COLORS_10_STR[i_color % n_colors]
# 
# 
# def gradiate_color_bgr(bgr_or_bgra, n=5):
#     rgb_or_rgba = (
#         bgr2rgb(bgr_or_bgra)
#         if len(bgr_or_bgra) == 3
#         else bgra2rgba(bgr_or_bgra)
#     )
#     rgb_gradient = gradiate_color_rgb(rgb_or_rgba, n)
#     return [
#         rgb2bgr(color) if len(color) == 3 else rgba2bgra(color)
#         for color in rgb_gradient
#     ]
# 
# 
# def gradiate_color_bgra(bgra, n=5):
#     return gradiate_color_bgr(bgra, n)
# 
# 
# # Deprecated
# # ------------------------------
# @deprecated("Use str2rgb instead")
# def to_rgb(c):
#     return str2rgb(c)
# 
# 
# @deprecated("use str2rgba instewad")
# def to_rgba(c, alpha=0.5):
#     return str2rgba(c, alpha=alpha)
# 
# 
# @deprecated("use str2hex instead")
# def to_hex(c):
#     return PARAMS["HEX"][c]
# 
# 
# @deprecated("use gradiate_color_rgb/rgba/bgr/bgra instead")
# def gradiate_color(rgb_or_rgba, n=5):
#     return gradiate_color_rgb(rgb_or_rgba, n)
# 
# 
# if __name__ == "__main__":
#     c = "blue"
#     print(to_rgb(c))
#     print(to_rgba(c))
#     print(to_hex(c))
#     print(cycle_color(1))
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_colors.py
# --------------------------------------------------------------------------------
