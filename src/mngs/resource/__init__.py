#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 10:41:33 (ywatanabe)"
# File: ./mngs_repo/src/mngs/resource/__init__.py

#!/usr/bin/env python3

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

# # import warnings

# # try:
# #     from ._get_proc_usages import get_proc_usages
# # except ImportError as e:
# #     warnings.warn(f"Warning: Failed to import get_proc_usages from ._get_proc_usages.")

# # try:
# #     from ._get_specs import get_specs
# # except ImportError as e:
# #     warnings.warn(f"Warning: Failed to import get_specs from ._get_specs.")

# # try:
# #     from ._rec_procs import rec_procs
# # except ImportError as e:
# #     warnings.warn(f"Warning: Failed to import rec_procs from ._rec_procs.")

# # try:
# #     from . import _utils
# # except ImportError as e:
# #     warnings.warn(f"Warning: Failed to import _utils.")


# from ._get_proc_usages import get_proc_usages
# from ._get_specs import get_specs
# from ._rec_procs import rec_procs
# from . import _utils

# _ = None  # keep the importing orders

# # from .limit_RAM import get_RAM, limit_RAM

# # from .get import system_info

# #

# EOF
