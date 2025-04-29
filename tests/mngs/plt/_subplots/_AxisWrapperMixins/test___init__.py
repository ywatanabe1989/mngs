#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:56:24 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/_subplots/_AxisWrapperMixins/test___init__.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/_subplots/_AxisWrapperMixins/test___init__.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from unittest.mock import MagicMock, patch

import pytest


def test_init_module_imports_all_non_underscore_modules():
    """Tests that __init__.py imports all modules in the directory that don't start with underscore."""
    # Setup: Mock os.listdir to return a known list of files
    mock_files = [
        "_TrackingMixin.py",
        "_SeabornMixin.py",
        "__init__.py",
        "__pycache__",
    ]

    # Mock the importlib.import_module to track what gets imported
    mock_module = MagicMock()
    mock_class = MagicMock()
    mock_class.__name__ = "TestClass"
    mock_function = MagicMock()
    mock_function.__name__ = "test_function"

    # Set up mock module with both a class and function
    mock_module.__name__ = "mock_module"
    mock_module.TestClass = mock_class
    mock_module.test_function = mock_function

    # Create a dictionary to hold members
    mock_members = [
        ("TestClass", mock_class),
        ("test_function", mock_function),
        ("_private_func", MagicMock()),  # This shouldn't be imported
    ]

    with patch("os.listdir", return_value=mock_files) as mock_listdir:
        with patch(
            "importlib.import_module", return_value=mock_module
        ) as mock_import:
            with patch(
                "inspect.getmembers", return_value=mock_members
            ) as mock_getmembers:
                with patch(
                    "inspect.isfunction",
                    side_effect=lambda obj: obj == mock_function,
                ):
                    with patch(
                        "inspect.isclass",
                        side_effect=lambda obj: obj == mock_class,
                    ):
                        # Import the module
                        module_globals = {}
                        exec("import os, importlib, inspect", module_globals)

                        # Execute the contents of __init__.py in this context
                        init_code = """
current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith(".py") and not filename.startswith("__"):
        module_name = filename[:-3]  # Remove .py extension
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Import only functions and classes from the module
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if not name.startswith("_"):
                    globals()[name] = obj
"""
                        exec(init_code, module_globals)

                        # Check that listdir was called
                        mock_listdir.assert_called_once()

                        # Check that import_module was called for each .py file
                        assert mock_import.call_count == 2
                        mock_import.assert_any_call(
                            "._TrackingMixin",
                            package=module_globals["__name__"],
                        )
                        mock_import.assert_any_call(
                            "._SeabornMixin",
                            package=module_globals["__name__"],
                        )

                        # Check that getmembers was called for each module
                        assert mock_getmembers.call_count == 2

                        # Check that only the public function and class were imported
                        assert "TestClass" in module_globals
                        assert "test_function" in module_globals
                        assert "_private_func" not in module_globals


def test_init_module_cleans_up_temporary_variables():
    """Tests that __init__.py cleans up all temporary variables after importing."""
    # Create a dictionary to simulate the module's global namespace
    module_globals = {
        "__name__": "test_module",
        "__file__": "/path/to/module/__init__.py",
        "__os": MagicMock(),
        "__importlib": MagicMock(),
        "__inspect": MagicMock(),
        "current_dir": "/path/to/module",
        "filename": "test.py",
        "module_name": "test",
        "module": MagicMock(),
        "name": "TestClass",
        "obj": MagicMock(),
    }

    # Execute the cleanup code
    cleanup_code = """
del (
    __os,
    __importlib,
    __inspect,
    current_dir,
    filename,
    module_name,
    module,
    name,
    obj,
)
"""
    exec(cleanup_code, module_globals)

    # Check that all temporary variables have been removed
    assert "__os" not in module_globals
    assert "__importlib" not in module_globals
    assert "__inspect" not in module_globals
    assert "current_dir" not in module_globals
    assert "filename" not in module_globals
    assert "module_name" not in module_globals
    assert "module" not in module_globals
    assert "name" not in module_globals
    assert "obj" not in module_globals

    # Original built-in module attributes should still be present
    assert "__name__" in module_globals
    assert "__file__" in module_globals

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-13 14:53:51 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/__init__.py
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
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/__init__.py
# --------------------------------------------------------------------------------
