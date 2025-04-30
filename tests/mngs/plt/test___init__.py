# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-29 20:41:53 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/__init__.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/__init__.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
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
# from ._subplots._SubplotsWrapper import subplots
# from ._PARAMS import PARAMS
# from . import ax
# from ._close import close
# 
# ################################################################################
# # For Matplotlib Compatibility
# ################################################################################
# import matplotlib.pyplot as _counter_part
# _local_module_attributes = list(globals().keys())
# 
# def __getattr__(name):
#     """
#     Fallback to fetch attributes from matplotlib.pyplot
#     if they are not defined directly in this module.
#     """
#     try:
#         # Get the attribute from matplotlib.pyplot
#         return getattr(_counter_part, name)
#     except AttributeError:
#         # Raise the standard error if not found in pyplot either
#         raise AttributeError(
#             f"module '{__name__}' nor matplotlib.pyplot has attribute '{name}'"
#         ) from None
# 
# def __dir__():
#     """
#     Provide combined directory for tab completion, including
#     attributes from this module and matplotlib.pyplot.
#     """
#     # Get attributes defined explicitly in this module
#     local_attrs = set(_local_module_attributes)
#     # Get attributes from matplotlib.pyplot
#     pyplot_attrs = set(dir(_counter_part))
#     # Return the sorted union
#     return sorted(local_attrs.union(pyplot_attrs))
# 
# """
# import matplotlib.pyplot as _counter_part
# import mngs.plt as mplt
# 
# set(dir(mplt)) - set(dir(_counter_part))
# set(dir(_counter_part))
# 
# mplt.yticks
# 
# is_compatible = np.all([kk in set(dir(mplt)) for kk in set(dir(_counter_part))])
# if is_compatible:
#     print(f"{mplt.__name__} is compatible with {_counter_part.__name__}")
# else:
#     print(f"{mplt.__name__} is incompatible with {_counter_part.__name__}")
# """
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/__init__.py
# --------------------------------------------------------------------------------
