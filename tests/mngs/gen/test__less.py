# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-03 02:11:18 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/gen/_less.py
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-21 12:05:35"
# # Author: Yusuke Watanabe (ywata1989@gmail.com)
# 
# """
# This script does XYZ.
# """
# 
# import sys
# 
# import matplotlib.pyplot as plt
# import mngs
# 
# # Imports
# 
# # # Config
# # CONFIG = mngs.gen.load_configs()
# 
# # Functions
# def less(output):
#     """
#     Print the given output using `less` in an IPython or IPdb session.
#     """
#     import os
#     import tempfile
# 
#     from IPython import get_ipython
# 
#     # Create a temporary file to hold the output
#     with tempfile.NamedTemporaryFile(delete=False, mode="w+t") as tmpfile:
#         # Write the output to the temporary file
#         tmpfile.write(output)
#         tmpfile_name = tmpfile.name
# 
#     # Use IPython's system command access to pipe the content of the temporary file to `less`
#     get_ipython().system(f"less {tmpfile_name}")
# 
#     # Clean up the temporary file
#     os.remove(tmpfile_name)
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

from src.mngs.gen._less import *

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