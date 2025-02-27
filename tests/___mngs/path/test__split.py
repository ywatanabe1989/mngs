# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/path/_split.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 16:18:06 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/path/_split.py
# 
# import os
# 
# def split(fpath):
#     """Split a file path to (1) the directory path, (2) the file name, and (3) the file extention
#     Example:
#         dirname, fname, ext = split('../data/01/day1/split_octave/2kHz_mat/tt8-2.mat')
#         print(dirname) # '../data/01/day1/split_octave/2kHz_mat/'
#         print(fname) # 'tt8-2'
#         print(ext) # '.mat'
#     """
#     dirname = os.path.dirname(fpath) + "/"
#     base = os.path.basename(fpath)
#     fname, ext = os.path.splitext(base)
#     return dirname, fname, ext
# 
# # EOF

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.path._split import *

class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        raise NotImplementedError("Test not yet implemented")

    def test_edge_cases(self):
        # Edge case testing
        raise NotImplementedError("Test not yet implemented")

    def test_error_handling(self):
        # Error handling testing
        raise NotImplementedError("Test not yet implemented")

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
