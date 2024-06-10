#!/usr/bin/env python3

# from . import colors
from ._PARAMS import PARAMS

_ = None
from . import ax
from ._add_hue import add_hue
from ._annotated_heatmap import annotated_heatmap
from ._colors import (
    cycle_color,
    rgb2rgba,
    rgba2hex,
    to_hex,
    to_rgb,
    to_rgba,
    update_alpha,
)
from ._configure_mpl import configure_mpl
from ._draw_a_cube import draw_a_cube
from ._get_RGBA_from_colormap import get_RGBA_from_colormap
from ._grid_image import grid_image
from ._interp_colors import interp_colors
from ._mk_colorbar import mk_colorbar
from ._mk_patches import mk_patches
from ._subplots._SubplotsManager import subplots
from ._tpl import termplot
