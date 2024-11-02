#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 16:08:50 (ywatanabe)"
# File: ./mngs_repo/src/mngs/typing/__init__.py

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

# EOF

from typing import *
from ._ArrayLike import ArrayLike

# from ._typing import List, Tuple, Dict, Any, Union, Sequence, Literal, Optional, Iterable, Generator, ArrayLike


# EOF
