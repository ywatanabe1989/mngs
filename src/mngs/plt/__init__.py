#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 12:50:28 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/__init__.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/__init__.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# import importlib
# import inspect

# # Get the current directory
# current_dir = os.path.dirname(__file__)

# # Iterate through all Python files in the current directory
# for filename in os.listdir(current_dir):
#     if filename.endswith(".py") and not filename.startswith("__"):
#         module_name = filename[:-3]  # Remove .py extension
#         module = importlib.import_module(f".{module_name}", package=__name__)

#         # Import only functions and classes from the module
#         for name, obj in inspect.getmembers(module):
#             if inspect.isfunction(obj) or inspect.isclass(obj):
#                 if not name.startswith("_"):
#                     globals()[name] = obj

# # Clean up temporary variables
# del os, importlib, inspect, current_dir, filename, module_name, module, name, obj
# from ._subplots._SubplotsManager import subplots
# from ._PARAMS import PARAMS
# from . import ax

from ._PARAMS import PARAMS
from ._configure_mpl import configure_mpl
from ._subplots._SubplotsManager import subplots
from . import ax
from ._colors import (
   to_rgb,
   to_rgba,
   to_hex,
   rgb2rgba,
   rgba2rgb,
   update_alpha,
   rgba2hex,
   cycle_color,
   gradiate_color,
)

# EOF