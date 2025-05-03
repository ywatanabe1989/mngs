#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 00:51:04 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/color/__init__.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/color/__init__.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from ._get_colors_from_cmap import (
    get_color_from_cmap,
    get_colors_from_cmap,
    get_categorical_colors_from_cmap,
)
from ._interpolate import interpolate
from ._PARAMS import PARAMS
from ._vizualize_colors import vizualize_colors
from ._colors import (
    # RGB
    str2rgb,
    str2rgba,
    rgb2rgba,
    rgba2rgb,
    rgba2hex,
    cycle_color_rgb,
    gradiate_color_rgb,
    gradiate_color_rgba,
    # BGR
    str2bgr,
    str2bgra,
    bgr2bgra,
    bgra2bgr,
    bgra2hex,
    cycle_color_bgr,
    gradiate_color_bgr,
    gradiate_color_bgra,
    # COMMON
    rgb2bgr,
    bgr2rgb,
    str2hex,
    update_alpha,
    cycle_color,
    gradiate_color,
    to_rgb,
    to_rgba,
    to_hex,
    gradiate_color,
)

# EOF