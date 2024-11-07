# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-06 09:00:45 (ywatanabe)"
# 
# # import os
# # import shutil
# 
# # def mv(src, tgt):
# #     successful = True
# #     os.makedirs(tgt, exist_ok=True)
# 
# #     if os.path.isdir(src):
# #         # Iterate over the items in the directory
# #         for item in os.listdir(src):
# #             item_path = os.path.join(src, item)
# #             # Check if the item is a file
# #             if os.path.isfile(item_path):
# #                 try:
# #                     shutil.move(item_path, tgt)
# #                     print(f"\nMoved file from {item_path} to {tgt}")
# #                 except OSError as e:
# #                     print(f"\nError: {e}")
# #                     successful = False
# #             else:
# #                 print(f"\nSkipped directory {item_path}")
# #     else:
# #         # If src is a file, just move it
# #         try:
# #             shutil.move(src, tgt)
# #             print(f"\nMoved from {src} to {tgt}")
# #         except OSError as e:
# #             print(f"\nError: {e}")
# #             successful = False
# 
# #     return successful
# 
# 
# def mv(src, tgt):
#     import os
#     import shutil
# 
#     successful = True
#     os.makedirs(tgt, exist_ok=True)
# 
#     try:
#         shutil.move(src, tgt)
#         print(f"\nMoved from {src} to {tgt}")
#     except OSError as e:
#         print(f"\nError: {e}")
#         successful = False

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

from mngs.os._mv import *

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
