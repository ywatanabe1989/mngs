#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-29 10:20:25 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/__init__.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/ax/__init__.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# Time-stamp: "2024-11-16 19:56:23 (ywatanabe)"
# File: ./mngs_repo/src/mngs/plt/ax/__init__.py

THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/__init__.py"

from ._add_marginal_ax import add_marginal_ax
from ._circular_hist import circular_hist
from ._ecdf import ecdf
from ._extend import extend
from ._fillv import fillv
from ._force_aspect import force_aspect
from ._format_label import format_label
from ._hide_spines import hide_spines
from ._imshow2d import imshow2d
from ._joyplot import joyplot
from ._map_ticks import map_ticks
from ._panel import panel
# from ._plot_ import fill_between, plot_, plot_with_ci
from ._plot_ import plot_
from ._raster_plot import raster_plot
from ._rectangle import rectangle
from ._sci_note import sci_note
from ._set_n_ticks import set_n_ticks
from ._set_size import set_size
from ._set_supxyt import set_supxyt
from ._set_ticks import set_ticks
from ._set_xyt import set_xyt
from ._share_axes import (get_global_xlim, get_global_ylim, set_xlims,
                          set_ylims, sharex, sharexy, sharey)
# from ._set_pos import set_pos
from ._shift import shift
from ._rotate_labels import rotate_labels
from ._half_violin import half_violin




# ################################################################################
# # For Matplotlib Compatibility
# ################################################################################
# import matplotlib.pyplot.axis as counter_part
# _local_module_attributes = list(globals().keys())
# print(_local_module_attributes)

# def __getattr__(name):
#     """
#     Fallback to fetch attributes from matplotlib.pyplot
#     if they are not defined directly in this module.
#     """
#     try:
#         # Get the attribute from matplotlib.pyplot
#         return getattr(counter_part, name)
#     except AttributeError:
#         # Raise the standard error if not found in pyplot either
#         raise AttributeError(
#             f"module '{__name__}' nor matplotlib.pyplot has attribute '{name}'"
#         ) from None

# def __dir__():
#     """
#     Provide combined directory for tab completion, including
#     attributes from this module and matplotlib.pyplot.
#     """
#     # Get attributes defined explicitly in this module
#     local_attrs = set(_local_module_attributes)
#     # Get attributes from matplotlib.pyplot
#     pyplot_attrs = set(dir(counter_part))
#     # Return the sorted union
#     return sorted(local_attrs.union(pyplot_attrs))

# """
# import matplotlib.pyplot as plt
# import mngs.plt as mplt

# print(set(dir(mplt.ax)) - set(dir(plt.axis)))
# """

# EOF