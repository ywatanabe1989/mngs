#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-26 08:57:13 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/__init__.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/decorators/__init__.py"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 13:05:15 (ywatanabe)"
# File: ./mngs_repo/src/mngs/decorators/__init__.py

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

# from ._DataTypeDecorators import (
#     torch_fn,
#     numpy_fn,
#     pandas_fn,
#     xarray_fn,
#     batch_fn,
#     batch_torch_fn,
#     batch_numpy_fn,
#     batch_pandas_fn,
#     batch_xarray_fn,
# )


# EOF
