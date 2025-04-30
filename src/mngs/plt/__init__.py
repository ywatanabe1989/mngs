#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 21:16:29 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/__init__.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/__init__.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from ._subplots._SubplotsWrapper import subplots
from ._PARAMS import PARAMS
from . import ax
from ._close import close
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
    to_rgb, to_rgba, to_hex, gradiate_color
)
from ._configure_mpl import configure_mpl
from ._im2grid import im2grid



################################################################################
# For Matplotlib Compatibility
################################################################################
import matplotlib.pyplot as _counter_part

_local_module_attributes = list(globals().keys())


def __getattr__(name):
    """
    Fallback to fetch attributes from matplotlib.pyplot
    if they are not defined directly in this module.
    """
    try:
        # Get the attribute from matplotlib.pyplot
        return getattr(_counter_part, name)
    except AttributeError:
        # Raise the standard error if not found in pyplot either
        raise AttributeError(
            f"module '{__name__}' nor matplotlib.pyplot has attribute '{name}'"
        ) from None


def __dir__():
    """
    Provide combined directory for tab completion, including
    attributes from this module and matplotlib.pyplot.
    """
    # Get attributes defined explicitly in this module
    local_attrs = set(_local_module_attributes)
    # Get attributes from matplotlib.pyplot
    pyplot_attrs = set(dir(_counter_part))
    # Return the sorted union
    return sorted(local_attrs.union(pyplot_attrs))


"""
import matplotlib.pyplot as _counter_part
import mngs.plt as mplt

set(dir(mplt)) - set(dir(_counter_part))
set(dir(_counter_part))

mplt.yticks

is_compatible = np.all([kk in set(dir(mplt)) for kk in set(dir(_counter_part))])
if is_compatible:
    print(f"{mplt.__name__} is compatible with {_counter_part.__name__}")
else:
    print(f"{mplt.__name__} is incompatible with {_counter_part.__name__}")
"""

# EOF