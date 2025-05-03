# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/io/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-22 09:27:44 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/io/__init__.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "/ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/io/__init__.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/io/__init__.py"
# 
# # import os
# # import importlib
# # import inspect
# 
# # # Get the current directory
# # current_dir = os.path.dirname(__file__)
# 
# # # Iterate through all Python files in the current directory
# # for filename in os.listdir(current_dir):
# #     if filename.endswith(".py") and not filename.startswith("__"):
# #         module_name = filename[:-3]  # Remove .py extension
# #         module = importlib.import_module(f".{module_name}", package=__name__)
# #         # Import only functions and classes from the module
# #         for name, obj in inspect.getmembers(module):
# #             if inspect.isfunction(obj) or inspect.isclass(obj):
# #                 if not name.startswith("_"):
# #                     # print(name)
# #                     globals()[name] = obj
# 
# # # Clean up temporary variables
# # del os, importlib, inspect, current_dir, filename, module_name, module, name, obj
# 
# # # EOF
# 
# from ._cache import *
# from ._flush import *
# from ._glob import *
# from ._json2md import *
# from ._load_configs import *
# from ._load_modules import *
# from ._load import *
# from ._mv_to_tmp import *
# from ._path import *
# from ._reload import *
# from ._save import *
# from ._save_image import *
# from ._save_listed_dfs_as_csv import *
# from ._save_listed_scalars_as_csv import *
# from ._save_mp4 import *
# from ._save_optuna_study_as_csv_and_pngs import *
# # from ._save_optuna_stury import *
# from ._save import *
# from ._save_text import *
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/io/__init__.py
# --------------------------------------------------------------------------------
