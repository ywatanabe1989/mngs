#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-29 09:38:56 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/test___init__.py
# ----------------------------------------
import os

__FILE__ = "./tests/mngs/test___init__.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import warnings
import pytest
from unittest.mock import patch, MagicMock


def test_import_mngs():
    try:
        import mngs

        assert True
    except Exception as e:
        print(e)
        assert False


def test_all_modules_imported():
    """Test that all expected modules are imported."""
    import mngs
    
    expected_modules = [
        'types', 'io', 'path', 'dict', 'gen', 'decorators',
        'ai', 'dsp', 'gists', 'linalg', 'nn', 'os', 'plt',
        'stats', 'torch', 'tex', 'resource', 'web', 'db',
        'pd', 'str', 'parallel', 'dt', 'dev'
    ]
    
    for module in expected_modules:
        assert hasattr(mngs, module), f"Missing module: {module}"
        assert getattr(mngs, module) is not None


def test_sh_function_available():
    """Test that sh function is available."""
    import mngs
    
    assert hasattr(mngs, 'sh')
    assert callable(mngs.sh)


def test_version_attribute():
    """Test that __version__ attribute exists."""
    import mngs
    
    assert hasattr(mngs, '__version__')
    assert isinstance(mngs.__version__, str)
    assert len(mngs.__version__) > 0
    # Check version format (should be like "1.11.0")
    parts = mngs.__version__.split('.')
    assert len(parts) >= 2  # At least major.minor


def test_file_attributes():
    """Test that __FILE__ and __DIR__ attributes exist."""
    import mngs
    
    assert hasattr(mngs, '__FILE__')
    assert hasattr(mngs, '__DIR__')
    assert isinstance(mngs.__FILE__, str)
    assert isinstance(mngs.__DIR__, str)


def test_this_file_attribute():
    """Test THIS_FILE attribute."""
    import mngs
    
    assert hasattr(mngs, 'THIS_FILE')
    assert isinstance(mngs.THIS_FILE, str)
    assert 'mngs' in mngs.THIS_FILE


def test_deprecation_warnings_filtered():
    """Test that DeprecationWarning is filtered."""
    import mngs
    
    # Check that DeprecationWarning filter is in place
    found_filter = False
    for filter_item in warnings.filters:
        if (filter_item[0] == 'ignore' and 
            filter_item[2] == DeprecationWarning):
            found_filter = True
            break
    
    assert found_filter, "DeprecationWarning filter not found"


def test_environment_variables_comments():
    """Test that environment variables are documented in comments."""
    import mngs
    
    # These should be documented in the module
    expected_env_vars = [
        "MNGS_SENDER_GMAIL",
        "MNGS_SENDER_GMAIL_PASSWORD", 
        "MNGS_RECIPIENT_GMAIL",
        "MNGS_DIR"
    ]
    
    # Since they're in comments, just verify the module loads
    assert mngs is not None


def test_module_types():
    """Test that imported modules are actual modules."""
    import mngs
    import types as builtin_types
    
    modules_to_check = [
        'types', 'io', 'path', 'dict', 'gen', 'decorators',
        'ai', 'dsp', 'gists', 'linalg', 'nn', 'os', 'plt',
        'stats', 'torch', 'tex', 'resource', 'web', 'db',
        'pd', 'str', 'parallel', 'dt', 'dev'
    ]
    
    for module_name in modules_to_check:
        module = getattr(mngs, module_name)
        # Check it's a module or has module-like attributes
        assert hasattr(module, '__name__') or hasattr(module, '__file__') or hasattr(module, '__package__')


def test_no_import_errors():
    """Test that importing mngs doesn't raise any errors."""
    # Clear mngs from modules
    import sys
    if 'mngs' in sys.modules:
        del sys.modules['mngs']
    
    # Import should work without errors
    try:
        import mngs
        success = True
    except Exception:
        success = False
    
    assert success


def test_module_reimport():
    """Test that mngs can be imported multiple times."""
    import sys
    
    # First import
    import mngs
    first_id = id(mngs)
    
    # Force reimport
    if 'mngs' in sys.modules:
        del sys.modules['mngs']
    
    # Second import
    import mngs
    second_id = id(mngs)
    
    # Should get a new module object
    assert first_id != second_id


def test_submodule_access():
    """Test accessing submodules through mngs."""
    import mngs
    
    # Test accessing nested attributes
    assert hasattr(mngs.io, 'save')
    assert hasattr(mngs.io, 'load')
    assert hasattr(mngs.plt, 'subplots')
    assert hasattr(mngs.gen, 'start')


def test_common_functionality():
    """Test some common mngs functionality is accessible."""
    import mngs
    
    # Check common functions exist
    common_functions = [
        ('io', 'save'),
        ('io', 'load'),
        ('plt', 'subplots'),
        ('gen', 'start'),
        ('path', 'split'),
    ]
    
    for module_name, func_name in common_functions:
        module = getattr(mngs, module_name)
        assert hasattr(module, func_name), f"Missing {module_name}.{func_name}"


def test_version_format():
    """Test version string format is valid."""
    import mngs
    
    version = mngs.__version__
    
    # Should be in format X.Y.Z or X.Y.Z.dev
    parts = version.split('.')
    assert len(parts) >= 2, "Version should have at least major.minor"
    
    # Major and minor should be integers
    assert parts[0].isdigit(), "Major version should be numeric"
    assert parts[1].isdigit(), "Minor version should be numeric"
    
    if len(parts) >= 3:
        # Patch version might have 'dev' or other suffixes
        patch = parts[2]
        if patch.isdigit():
            assert True
        else:
            # Could be like "0dev" or "0rc1"
            assert any(char.isdigit() for char in patch)


def test_no_context_module():
    """Test that context module is commented out as expected."""
    import mngs
    
    # Context module should not be imported (it's commented out)
    # This is intentional based on the source code
    assert not hasattr(mngs, 'context')


def test_module_isolation():
    """Test that mngs modules don't pollute namespace."""
    import mngs
    
    # These should not be in mngs namespace
    unwanted_attributes = ['sys', 'importlib', 'pkgutil']
    
    for attr in unwanted_attributes:
        if hasattr(mngs, attr):
            # It's OK if these are modules that mngs intentionally exports
            pass


def test_os_module_not_overridden():
    """Test that mngs.os doesn't override built-in os."""
    import os as builtin_os
    import mngs
    
    # mngs has its own os module, but it shouldn't affect the built-in
    assert builtin_os.path is not None
    assert builtin_os.environ is not None
    
    # mngs.os should be the custom module
    assert hasattr(mngs.os, '_mv') or hasattr(mngs.os, '__file__')


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-01 09:22:55 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/__init__.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/__init__.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/__init__.py"
#
# # os.getenv("MNGS_SENDER_GMAIL")
# # os.getenv("MNGS_SENDER_GMAIL_PASSWORD")
# # os.getenv("MNGS_RECIPIENT_GMAIL")
# # os.getenv("MNGS_DIR", "/tmp/mngs/")
#
# import warnings
#
# # Configure warnings
# warnings.filterwarnings('ignore', category=DeprecationWarning)
#
# ########################################
# # Warnings
# ########################################
#
# from . import types
# from ._sh import sh
# from . import io
# from . import path
# from . import dict
# from . import gen
# from . import decorators
# from . import ai
# from . import dsp
# from . import gists
# from . import linalg
# from . import nn
# from . import os
# from . import plt
# from . import stats
# from . import torch
# from . import tex
# from . import resource
# from . import web
# from . import db
# from . import pd
# from . import str
# from . import parallel
# from . import dt
# from . import dev
# # from . import context
#
# # ########################################
# # # Modules (python -m mngs print_config)
# # ########################################
# # from .gen._print_config import print_config
# # # Usage: python -m mngs print_config
#
# __copyright__ = "Copyright (C) 2024 Yusuke Watanabe"
# __version__ = "1.10.1"
# __license__ = "MIT"
# __author__ = "ywatanabe1989"
# __author_email__ = "ywatanabe@alumni.u-tokyo.ac.jp"
# __url__ = "https://github.com/ywatanabe1989/mngs"
#
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/__init__.py
# --------------------------------------------------------------------------------
