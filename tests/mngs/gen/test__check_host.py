# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 13:43:36 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/gen/_check_host.py
# 
# 
# from .._sh import sh
# import sys
# 
# def check_host(keyword):
#     return keyword in sh('echo $(hostname)', verbose=False)
# 
# is_host = check_host
# 
# def verify_host(keyword):
#     if is_host(keyword):
#         print(f"Host verification successed for keyword: {keyword}")
#         return
#     else:
#         print(f"Host verification failed for keyword: {keyword}")
#         sys.exit(1)
# 
# if __name__ == '__main__':
#     # check_host("ywata")
#     verify_host("titan")
#     verify_host("ywata")
#     verify_host("crest")
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

from src.mngs.gen._check_host import *

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
