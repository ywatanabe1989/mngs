#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-24 18:40:30 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/pd/__init__.py

import os
import importlib
import inspect
import warnings

# Get the current directory
current_dir = os.path.dirname(__file__)

# Iterate through all Python files in the current directory
for filename in os.listdir(current_dir):
    if filename.endswith(".py") and not filename.startswith("__"):
        module_name = filename[:-3]
        # module = importlib.import_module(f".{module_name}", package=__name__)

        # # Import only functions and classes from the module
        # for name, obj in inspect.getmembers(module):
        #     if inspect.isfunction(obj) or inspect.isclass(obj):
        #         if not name.startswith("_"):
        #             globals()[name] = obj
        try:
            module = importlib.import_module(f".{module_name}", package=__name__)

            # Import only functions and classes from the module
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) or inspect.isclass(obj):
                    if not name.startswith("_"):
                        globals()[name] = obj
        except ImportError as e:
            warnings.warn(f"Warning: Failed to import {module_name}.")

# Clean up temporary variables
del (
    os,
    importlib,
    inspect,
    current_dir,
    filename,
    module_name,
    module,
    name,
    obj,
)

# from ._merge_columns import merge_cols, merge_columns
# from ._misc import find_indi  # col_to_last,; col_to_top,; merge_columns,
# from ._misc import force_df, ignore_SettingWithCopyWarning, slice
# from ._mv import mv, mv_to_first, mv_to_last
# from ._sort import sort
