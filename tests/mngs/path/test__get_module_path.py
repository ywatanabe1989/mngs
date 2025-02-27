# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:39:32 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/path/_get_module_path.py
# 
# def get_data_path_from_a_package(package_str, resource):
#     """
#     Get the path to a data file within a package.
# 
#     This function finds the path to a data file within a package's data directory.
# 
#     Parameters:
#     -----------
#     package_str : str
#         The name of the package as a string.
#     resource : str
#         The name of the resource file within the package's data directory.
# 
#     Returns:
#     --------
#     str
#         The full path to the resource file.
# 
#     Raises:
#     -------
#     ImportError
#         If the specified package cannot be found.
#     FileNotFoundError
#         If the resource file does not exist in the package's data directory.
#     """
#     import importlib
#     import os
#     import sys
# 
#     spec = importlib.util.find_spec(package_str)
#     if spec is None:
#         raise ImportError(f"Package '{package_str}' not found")
# 
#     data_dir = os.path.join(spec.origin.split("src")[0], "data")
#     resource_path = os.path.join(data_dir, resource)
# 
#     if not os.path.exists(resource_path):
#         raise FileNotFoundError(
#             f"Resource '{resource}' not found in package '{package_str}'"
#         )
# 
#     return resource_path
# 
# 
# # EOF

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs..path._get_module_path import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        pass

    def test_edge_cases(self):
        # Edge case testing
        pass

    def test_error_handling(self):
        # Error handling testing
        pass
