# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 20:59:46 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/path/_mk_spath.py
# 
# import inspect
# import os
# 
# from ._split import split
# 
# 
# 
# def mk_spath(sfname, makedirs=False):
#     """
#     Create a save path based on the calling script's location.
# 
#     Parameters:
#     -----------
#     sfname : str
#         The name of the file to be saved.
#     makedirs : bool, optional
#         If True, create the directory structure for the save path. Default is False.
# 
#     Returns:
#     --------
#     str
#         The full save path for the file.
# 
#     Example:
#     --------
#     >>> import mngs.io._path as path
#     >>> spath = path.mk_spath('output.txt', makedirs=True)
#     >>> print(spath)
#     '/path/to/current/script/output.txt'
#     """
#     __file__ = inspect.stack()[1].filename
#     if "ipython" in __file__:  # for ipython
#         __file__ = f'/tmp/fake-{os.getenv("USER")}.py'
# 
#     ## spath
#     fpath = __file__
#     fdir, fname, _ = split(fpath)
#     sdir = fdir + fname + "/"
#     spath = sdir + sfname
# 
#     if makedirs:
#         os.makedirs(split(spath)[0], exist_ok=True)
# 
#     return spath
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
    sys.path.insert(0, project_root)

from src.mngs.path/_mk_spath.py import *

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
