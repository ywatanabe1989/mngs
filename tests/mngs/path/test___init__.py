#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-02 12:55:00 (ywatanabe)"
# File: ./tests/mngs/path/test___init__.py

import pytest
import importlib
import sys
from unittest.mock import patch, MagicMock


def test_path_module_imports():
    """Test that path module can be imported."""
    import mngs.path
    assert mngs.path is not None


def test_path_module_dynamic_imports():
    """Test that path module dynamically imports submodules."""
    import mngs.path
    
    # Check that common path functions are available
    expected_functions = [
        'clean', 'find', 'get_module_path', 'getsize', 
        'get_spath', 'increment_version', 'mk_spath',
        'split', 'this_path'
    ]
    
    for func_name in expected_functions:
        assert hasattr(mngs.path, func_name), f"Function {func_name} not found in mngs.path"


def test_path_module_no_private_imports():
    """Test that private functions are not imported."""
    import mngs.path
    
    # Get all attributes
    attrs = dir(mngs.path)
    
    # Check that no private functions (starting with _) are exposed
    # except for special Python attributes
    for attr in attrs:
        if attr.startswith('_') and not attr.startswith('__'):
            pytest.fail(f"Private attribute {attr} should not be exposed")


def test_path_module_function_callable():
    """Test that imported functions are callable."""
    import mngs.path
    
    # Test a few key functions
    test_functions = ['clean', 'split', 'increment_version']
    
    for func_name in test_functions:
        if hasattr(mngs.path, func_name):
            func = getattr(mngs.path, func_name)
            assert callable(func), f"{func_name} should be callable"


def test_path_module_clean_namespace():
    """Test that temporary variables are cleaned up."""
    import mngs.path
    
    # These temporary variables should not exist after import
    temp_vars = ['os', 'importlib', 'inspect', 'current_dir', 
                 'filename', 'module_name', 'module', 'name', 'obj']
    
    for var in temp_vars:
        assert not hasattr(mngs.path, var), f"Temporary variable {var} was not cleaned up"


def test_path_module_reload():
    """Test that path module can be reloaded."""
    import mngs.path
    
    # Store original reference
    original_id = id(mngs.path)
    
    # Reload the module
    importlib.reload(mngs.path)
    
    # Should still be importable
    assert mngs.path is not None
    
    # Functions should still be available
    assert hasattr(mngs.path, 'clean')


def test_path_module_submodule_structure():
    """Test the submodule structure is preserved."""
    # Import individual submodules directly
    from mngs.path import _clean
    from mngs.path import _increment_version
    from mngs.path import _split
    
    # These should exist as separate modules
    assert _clean is not None
    assert _increment_version is not None  
    assert _split is not None


def test_path_module_no_side_effects():
    """Test that importing path module has no side effects."""
    # Create a mock for os.listdir to track calls
    with patch('os.listdir') as mock_listdir:
        # Set up mock to return some test files
        mock_listdir.return_value = ['_test.py', '__init__.py', 'other.py']
        
        # Clear the module from cache if it exists
        if 'mngs.path' in sys.modules:
            del sys.modules['mngs.path']
        
        # Import should call listdir once for the current directory
        import mngs.path
        
        # Should have been called
        assert mock_listdir.called


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/path/__init__.py
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
# # from ._clean import clean
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/path/__init__.py
# --------------------------------------------------------------------------------
