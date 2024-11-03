#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 21:30:58 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/__init__.py

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
                    # print(name)
                    globals()[name] = obj

# Clean up temporary variables
del os, importlib, inspect, current_dir, filename, module_name, module, name, obj

# EOF

# from ._load import (
#     load_study_rdb,
#     load_yaml_as_an_optuna_dict,
# )
# # from ._glob import glob
# # from ._reload import reload
# from ._save import save_optuna_study_as_csv_and_pngs

# EOF
