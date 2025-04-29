#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:49:30 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/_subplots/test___init__.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/_subplots/test___init__.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import sys
from unittest.mock import patch


class TestSubplotsInit:
    def test_imports(self):
        with patch.dict(sys.modules):
            if "mngs.plt._subplots" in sys.modules:
                del sys.modules["mngs.plt._subplots"]

            # Import the module
            import mngs.plt._subplots

            # Check if important classes/functions are in the namespace
            assert hasattr(mngs.plt._subplots, "subplots")

    def test_dynamic_imports(self):
        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = ["_SubplotsManager.py", "__init__.py"]

            with patch.dict(sys.modules):
                if "mngs.plt._subplots" in sys.modules:
                    del sys.modules["mngs.plt._subplots"]

                with patch("importlib.import_module") as mock_import:
                    mock_module = type(
                        "MockModule", (), {"subplots": "mock_subplots"}
                    )
                    mock_import.return_value = mock_module

                    # Import the module
                    import mngs.plt._subplots

                    # Check that importlib.import_module was called
                    mock_import.assert_called()

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 00:55:34 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/plt/_subplots/__init__.py
# 
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-10 23:24:25 (ywatanabe)"
# # /home/ywatanabe/proj/mngs/src/mngs/plt/_subplots/__init__.py
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-22 19:51:47 (ywatanabe)"
# # File: __init__.py
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
# # from ._Subplot_manager import subplots
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/__init__.py
# --------------------------------------------------------------------------------
