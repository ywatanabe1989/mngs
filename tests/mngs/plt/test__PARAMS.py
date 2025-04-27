#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 16:38:46 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__PARAMS.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__PARAMS.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/_PARAMS.py
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

import sys

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.plt._PARAMS import PARAMS


def test_params_keys():
    expected = {"RGB", "RGBA", "RGBA_NORM", "RGBA_NORM_FOR_CYCLE", "HEX"}
    assert set(PARAMS.keys()) == expected


def test_rgb_value_red():
    rgb_red = PARAMS["RGB"]["red"]
    assert isinstance(rgb_red, list)
    assert rgb_red == [255, 70, 50]

# EOF