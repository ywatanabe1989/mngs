# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_load_modules/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:41:25 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_load_modules/__init__.py
# 
# import importlib as __importlib
# import inspect as __inspect
# import os as __os
# 
# # Get the current directory
# current_dir = __os.path.dirname(__file__)
# 
# # Iterate through all Python files in the current directory
# for filename in __os.listdir(current_dir):
#     if filename.endswith(".py") and not filename.startswith("__"):
#         module_name = filename[:-3]  # Remove .py extension
#         module = __importlib.import_module(f".{module_name}", package=__name__)
# 
#         # Import only functions and classes from the module
#         for name, obj in __inspect.getmembers(module):
#             if __inspect.isfunction(obj) or __inspect.isclass(obj):
#                 if not name.startswith("_"):
#                     globals()[name] = obj
# 
# # Clean up temporary variables
# del (
#     __os,
#     __importlib,
#     __inspect,
#     current_dir,
#     filename,
#     module_name,
#     module,
#     name,
#     obj,
# )
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_load_modules/__init__.py
# --------------------------------------------------------------------------------
