# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/path/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 21:00:41 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/path/__init__.py
# 
# import os
# import importlib
# import inspect
# 
# # Get the current directory
# current_dir = os.path.dirname(__file__)
# 
# # Iterate through all Python files in the current directory
# for filename in os.listdir(current_dir):
#     if filename.endswith(".py") and not filename.startswith("__"):
#         module_name = filename[:-3]  # Remove .py extension
#         module = importlib.import_module(f".{module_name}", package=__name__)
# 
#         # Import only functions and classes from the module
#         for name, obj in inspect.getmembers(module):
#             if inspect.isfunction(obj) or inspect.isclass(obj):
#                 if not name.startswith("_"):
#                     globals()[name] = obj
# 
# # Clean up temporary variables
# del os, importlib, inspect, current_dir, filename, module_name, module, name, obj
# 
# # from ._find import find_dir, find_file, find_git_root
# # from ._path import file_size, spath, split, this_path
# # from ._version import find_latest, increment_version
# from ._clean import clean
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/path/__init__.py
# --------------------------------------------------------------------------------
