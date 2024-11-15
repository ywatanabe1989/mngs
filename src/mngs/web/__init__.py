#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-11 06:58:21 (ywatanabe)"
# File: ./mngs_repo/src/mngs/web/__init__.py

#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-13 19:04:32 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/web/__init__.py

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

# EOF

# try:
#     from ._summarize_url import summarize_url
# except ImportError as e:
#     pass
#     # print(f"Warning: Failed to import summarize_url from ._summarize_url.")

# # from ._summarize_url import summarize_url


# EOF
