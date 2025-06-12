#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 11:53:18 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/__init__.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/plt/__init__.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from ._subplots._SubplotsWrapper import subplots
from . import ax
from . import color
from .utils._close import close
from ._tpl import tpl
from .utils._colorbar import colorbar as enhanced_colorbar

# Monkey patch matplotlib.pyplot.close to handle FigWrapper objects
import matplotlib.pyplot as _plt
_original_close = _plt.close

def _enhanced_close(fig=None):
    """Enhanced close that handles FigWrapper objects"""
    if fig is None:
        return _original_close()
    
    # Check if it's a FigWrapper object
    if hasattr(fig, '_fig_mpl') and hasattr(fig, 'figure'):
        # It's a FigWrapper, close the underlying figure
        return _original_close(fig.figure)
    else:
        # Regular matplotlib figure or other object
        return _original_close(fig)

# Replace matplotlib's close with our enhanced version
_plt.close = _enhanced_close

# Store original tight_layout before we define our enhanced version
_original_tight_layout = _plt.tight_layout

# Enhanced tight_layout that handles warnings
def tight_layout(*args, **kwargs):
    """Enhanced tight_layout that suppresses warnings about incompatible axes.
    
    This function wraps matplotlib.pyplot.tight_layout to handle cases where
    certain axes (like colorbars) are incompatible with tight_layout.
    If the current figure is using constrained_layout, this function does nothing.
    """
    import warnings
    
    # Get current figure
    fig = _plt.gcf()
    
    # Check if figure is already using constrained_layout
    if hasattr(fig, 'get_constrained_layout') and fig.get_constrained_layout():
        # Figure is using constrained_layout, which handles colorbars better
        # No need to call tight_layout
        return
    
    try:
        with warnings.catch_warnings():
            # Suppress the specific warning about incompatible axes
            warnings.filterwarnings("ignore", 
                                  message="This figure includes Axes that are not compatible with tight_layout")
            return _original_tight_layout(*args, **kwargs)
    except Exception:
        # If tight_layout fails, try to use constrained_layout on current figure
        try:
            fig.set_constrained_layout(True)
            fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04)
        except Exception:
            # If both fail, do nothing - figure will use default layout
            pass

# Replace matplotlib's tight_layout with our enhanced version
_plt.tight_layout = tight_layout


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
    # Special handling for close to use our enhanced version
    if name == 'close':
        return close
    
    # Special handling for tight_layout to use our enhanced version
    if name == 'tight_layout':
        return tight_layout
    
    # Special handling for colorbar to use our enhanced version
    if name == 'colorbar':
        return enhanced_colorbar
    
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
