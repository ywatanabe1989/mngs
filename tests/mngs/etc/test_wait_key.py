# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2023-03-24 23:13:32 (ywatanabe)"
# 
# import readchar
# import time
# import multiprocessing
# 
# def wait_key(p):
#     key = "x"
#     while key != "q":
#         key = readchar.readchar()
#         print(key)        
#     print("q was pressed.")
#     p.terminate()
#     # event.set()
#     # raise Exception
# 
# 
# def count():
#     counter = 0
#     while True:
#         print(counter)
#         time.sleep(1)
#         counter += 1
# 
# if __name__ == "__main__":
#     p1 = multiprocessing.Process(target=count)
#     
#     p1.start()
#     waitKey(p1)
#     print("aaa")

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

from src.mngs.etc/wait_key.py import *

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
