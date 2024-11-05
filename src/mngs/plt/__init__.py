#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-06 00:01:06 (ywatanabe)"
# File: ./mngs_repo/src/mngs/plt/__init__.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-22 19:51:47 (ywatanabe)"
# File: __init__.py

import os
import importlib
import inspect

# Get the current directory
current_dir = os.path.dirname(__file__)

# Iterate through all Python files in the current directory
for filename in os.listdir(current_dir):
    if filename.endswith(".py") and not filename.startswith("__"):
        module_name = filename[:-3]  # Remove .py extension
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Import only functions and classes from the module
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if not name.startswith("_"):
                    globals()[name] = obj

# Clean up temporary variables
del os, importlib, inspect, current_dir, filename, module_name, module, name, obj

from ._subplots._SubplotsManager import subplots
from ._PARAMS import PARAMS
from . import ax
# EOF

# #!/usr/bin/env python3

# try:
#     from ._PARAMS import PARAMS
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import PARAMS from ._PARAMS.")

# _ = None

# try:
#     from . import ax
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import ax.")

# try:
#     from ._add_hue import add_hue
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import add_hue from ._add_hue.")

# try:
#     from ._annotated_heatmap import annotated_heatmap
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import annotated_heatmap from ._annotated_heatmap.")

# try:
#     from ._colors import (
#         cycle_color,
#         rgb2rgba,
#         rgba2hex,
#         to_hex,
#         to_rgb,
#         to_rgba,
#         update_alpha,
#         gradiate_color,
#     )
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import from ._colors.")

# try:
#     from ._configure_mpl import configure_mpl
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import configure_mpl from ._configure_mpl.")

# try:
#     from ._draw_a_cube import draw_a_cube
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import draw_a_cube from ._draw_a_cube.")

# try:
#     from ._get_RGBA_from_colormap import get_RGBA_from_colormap
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import get_RGBA_from_colormap from ._get_RGBA_from_colormap.")

# try:
#     from ._grid_image import grid_image
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import grid_image from ._grid_image.")

# try:
#     from ._interp_colors import interp_colors
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import interp_colors from ._interp_colors.")

# try:
#     from ._mk_colorbar import mk_colorbar
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import mk_colorbar from ._mk_colorbar.")

# try:
#     from ._mk_patches import mk_patches
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import mk_patches from ._mk_patches.")

# try:
#     from ._subplots._SubplotsManager import subplots
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import subplots from ._subplots._SubplotsManager.")

# try:
#     from ._tpl import termplot
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import termplot from ._tpl.")

# # #!/usr/bin/env python3

# # # from . import colors
# # from ._PARAMS import PARAMS

# # _ = None
# # from . import ax
# # from ._add_hue import add_hue
# # from ._annotated_heatmap import annotated_heatmap
# # from ._colors import (
# #     cycle_color,
# #     rgb2rgba,
# #     rgba2hex,
# #     to_hex,
# #     to_rgb,
# #     to_rgba,
# #     update_alpha,
# #     gradiate_color,
# # )
# # from ._configure_mpl import configure_mpl
# # from ._draw_a_cube import draw_a_cube
# # from ._get_RGBA_from_colormap import get_RGBA_from_colormap
# # from ._grid_image import grid_image
# # from ._interp_colors import interp_colors
# # from ._mk_colorbar import mk_colorbar
# # from ._mk_patches import mk_patches
# # from ._subplots._SubplotsManager import subplots
# # from ._tpl import termplot


# EOF
