# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-25 00:31:08 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_glob.py
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/io/_glob.py"
# 
# import re as _re
# from glob import glob as _glob
# from ..str._parse import parse as _parse
# from natsort import natsorted as _natsorted
# 
# 
# def glob(expression, parse=False, ensure_one=False):
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
#     parse : bool, optional
#         Whether to parse the matched paths. Default is False.
#     ensure_one : bool, optional
#         Ensure exactly one match is found. Default is False.
# 
#     Returns:
#     --------
#     Union[List[str], Tuple[List[str], List[dict]]]
#         If parse=False: A naturally sorted list of file paths
#         If parse=True: Tuple of (paths, parsed results)
# 
#     Examples:
#     ---------
#     >>> glob('data/*.txt')
#     ['data/file1.txt', 'data/file2.txt', 'data/file10.txt']
# 
#     >>> glob('data/{a,b}/*.txt')
#     ['data/a/file1.txt', 'data/a/file2.txt', 'data/b/file1.txt']
# 
#     >>> paths, parsed = glob('data/subj_{id}/run_{run}.txt', parse=True)
#     >>> paths
#     ['data/subj_001/run_01.txt', 'data/subj_001/run_02.txt']
#     >>> parsed
#     [{'id': '001', 'run': '01'}, {'id': '001', 'run': '02'}]
# 
#     >>> paths, parsed = glob('data/subj_{id}/run_{run}.txt', parse=True, ensure_one=True)
#     AssertionError  # if more than one file matches
#     """
#     glob_pattern = _re.sub(r"{[^}]*}", "*", expression)
#     try:
#         found_paths = _natsorted(_glob(eval(glob_pattern)))
#     except:
#         found_paths = _natsorted(_glob(glob_pattern))
# 
#     if ensure_one:
#         assert len(found_paths) == 1
# 
#     if parse:
#         parsed = [_parse(found_path, expression) for found_path in found_paths]
#         return found_paths, parsed
# 
#     else:
#         return found_paths
# 
# def parse_glob(expression, ensure_one=False):
#     """
#     Convenience function for glob with parsing enabled.
# 
#     Parameters:
#     -----------
#     expression : str
#         The glob pattern to match against file paths.
#     ensure_one : bool, optional
#         Ensure exactly one match is found. Default is False.
# 
#     Returns:
#     --------
#     Tuple[List[str], List[dict]]
#         Matched paths and parsed results.
# 
#     Examples:
#     ---------
#     >>> paths, parsed = pglob('data/subj_{id}/run_{run}.txt')
#     >>> paths
#     ['data/subj_001/run_01.txt', 'data/subj_001/run_02.txt']
#     >>> parsed
#     [{'id': '001', 'run': '01'}, {'id': '001', 'run': '02'}]
# 
#     >>> paths, parsed = pglob('data/subj_{id}/run_{run}.txt', ensure_one=True)
#     AssertionError  # if more than one file matches
#     """
#     return glob(expression, parse=True, ensure_one=ensure_one)
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

from mngs..io._glob import *

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