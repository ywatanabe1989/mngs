# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-03 04:30:46 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_glob.py
# 
# import re
# from glob import glob as _glob
# 
# # from natsort import natsorted
# try:
#     from natsort import natsorted
# except ImportError as e:
#     import sys
#     print(f"Error importing natsort: {e}", file=sys.stderr)
#     print(f"Python path: {sys.path}", file=sys.stderr)
#     raise
# 
# 
# def glob(expression, ensure_one=False):
#     """
#     Perform a glob operation with natural sorting and extended pattern support.
# 
#     This function extends the standard glob functionality by adding natural sorting
#     and support for curly brace expansion in the glob pattern.
# 
#     Parameters:
#     -----------
#     expression : str
#         The glob pattern to match against file paths. Supports standard glob syntax
#         and curly brace expansion (e.g., 'dir/{a,b}/*.txt').
# 
#     Returns:
#     --------
#     list
#         A naturally sorted list of file paths that match the given expression.
# 
#     Examples:
#     ---------
#     >>> glob('data/*.txt')
#     ['data/file1.txt', 'data/file2.txt', 'data/file10.txt']
# 
#     >>> glob('data/{a,b}/*.txt')
#     ['data/a/file1.txt', 'data/a/file2.txt', 'data/b/file1.txt']
#     """
#     glob_pattern = re.sub(r"{[^}]*}", "*", expression)
#     try:
#         found_paths = natsorted(_glob(eval(glob_pattern)))
#     except:
#         found_paths = natsorted(_glob(glob_pattern))
# 
#     if ensure_one:
#         assert len(found_paths) == 1
# 
#     return found_paths
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

from mngs.io._glob import *

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
