#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 09:15:19 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__PARAMS.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__PARAMS.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest


def test_params_rgb_keys():
    """Test that RGB dictionary has expected keys."""
    from mngs.plt._PARAMS import RGB

    expected_keys = [
        "white",
        "black",
        "blue",
        "red",
        "pink",
        "green",
        "yellow",
        "gray",
        "grey",
        "purple",
        "light_blue",
        "brown",
        "navy",
        "orange",
    ]

    assert set(RGB.keys()) == set(expected_keys)


def test_params_rgb_values():
    """Test that RGB values are valid."""
    from mngs.plt._PARAMS import RGB

    for color, values in RGB.items():
        assert len(values) == 3, f"RGB color {color} should have 3 values"
        for value in values:
            assert (
                0 <= value <= 255
            ), f"RGB value for {color} should be between 0 and 255"


def test_params_rgba_norm():
    """Test that RGBA_NORM values are normalized correctly."""
    from mngs.plt._PARAMS import DEF_ALPHA, RGB, RGBA_NORM

    for color in RGB:
        rgb_values = RGB[color]
        rgba_norm_values = RGBA_NORM[color]

        assert (
            len(rgba_norm_values) == 4
        ), f"RGBA_NORM for {color} should have 4 values"

        for idx in range(3):
            expected = round(rgb_values[idx] / 255, 2)
            assert (
                rgba_norm_values[idx] == expected
            ), f"RGBA_NORM for {color} not correctly normalized"

        assert (
            rgba_norm_values[3] == DEF_ALPHA
        ), f"Alpha value for {color} is not set to DEF_ALPHA"

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_PARAMS.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-08 00:46:31 (ywatanabe)"
# # /home/ywatanabe/proj/mngs/src/mngs/plt/_PARAMS.py
# 
# import numpy as np
# 
# # RGB
# RGB = {
#     "white": [255, 255, 255],
#     "black": [0, 0, 0],
#     "blue": [0, 128, 192],
#     "red": [255, 70, 50],
#     "pink": [255, 150, 200],
#     "green": [20, 180, 20],
#     "yellow": [230, 160, 20],
#     "gray": [128, 128, 128],
#     "grey": [128, 128, 128],
#     "purple": [200, 50, 255],
#     "light_blue": [20, 200, 200],
#     "brown": [128, 0, 0],
#     "navy": [0, 0, 100],
#     "orange": [228, 94, 50],
# }
# 
# RGB_NORM = {
#     k: [round(r / 255, 2), round(g / 255, 2), round(b / 255, 2)]
#     for k, (r, g, b) in RGB.items()
# }
# 
# # RGBA
# DEF_ALPHA = 0.9
# RGBA = {k: [r, g, b, DEF_ALPHA] for k, (r, g, b) in RGB.items()}
# RGBA_NORM = {k: [r, g, b, DEF_ALPHA] for k, (r, g, b) in RGB_NORM.items()}
# RGBA_NORM_FOR_CYCLE = {
#     k: v for k, v in RGBA_NORM.items() if k not in ["white", "grey", "black"]
# }
# 
# # HEX
# HEX = {
#     "blue": "#0080C0",
#     "red": "#FF4632",
#     "pink": "#FF96C8",
#     "green": "#14B414",
#     "yellow": "#E6A014",
#     "gray": "#808080",
#     "grey": "#808080",
#     "purple": "#C832FF",
#     "light_blue": "#14C8C8",
#     "brown": "#800000",
#     "navy": "#000064",
#     "orange": "#E45E32",
# }
# 
# 
# PARAMS = dict(
#     RGB=RGB,
#     RGBA=RGBA,
#     RGBA_NORM=RGBA_NORM,
#     RGBA_NORM_FOR_CYCLE=RGBA_NORM_FOR_CYCLE,
#     HEX=HEX,
# )
# 
# # pprint(PARAMS)

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_PARAMS.py
# --------------------------------------------------------------------------------
