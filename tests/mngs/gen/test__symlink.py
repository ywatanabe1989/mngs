# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 13:29:31 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/gen/_symlink.py
# 
# import os
# from ..str._color_text import color_text
# 
# def symlink(tgt, src, force=False):
#     """Create a symbolic link.
# 
#     This function creates a symbolic link from the target to the source.
#     If the force parameter is True, it will remove any existing file at
#     the source path before creating the symlink.
# 
#     Parameters
#     ----------
#     tgt : str
#         The target path (the file or directory to be linked to).
#     src : str
#         The source path (where the symbolic link will be created).
#     force : bool, optional
#         If True, remove the existing file at the src path before creating
#         the symlink (default is False).
# 
#     Returns
#     -------
#     None
# 
#     Raises
#     ------
#     OSError
#         If the symlink creation fails.
# 
#     Example
#     -------
#     >>> symlink('/path/to/target', '/path/to/link')
#     >>> symlink('/path/to/target', '/path/to/existing_file', force=True)
#     """
#     if force:
#         try:
#             os.remove(src)
#         except FileNotFoundError:
#             pass
# 
#     # Calculate the relative path from src to tgt
#     src_dir = os.path.dirname(src)
#     relative_tgt = os.path.relpath(tgt, src_dir)
# 
#     os.symlink(relative_tgt, src)
#     print(
#         color_text(
#             f"\nSymlink was created: {src} -> {relative_tgt}\n", c="yellow"
#         )
#     )
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
    sys.path.insert(0, project_root)

from src.mngs.gen/_symlink.py import *

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
