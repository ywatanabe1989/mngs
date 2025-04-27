# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/path/_version.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 20:48:24 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/path/_version.py
# 
# import os
# import re
# import sys
# from glob import glob
# 
# import matplotlib.pyplot as plt
# 
# 
# # Functions
# def find_latest(dirname, fname, ext, version_prefix="_v"):
#     version_pattern = re.compile(
#         rf"({re.escape(fname)}{re.escape(version_prefix)})(\d+)({re.escape(ext)})$"
#     )
# 
#     glob_pattern = os.path.join(dirname, f"{fname}{version_prefix}*{ext}")
#     files = glob(glob_pattern)
# 
#     highest_version = 0
#     latest_file = None
# 
#     for file in files:
#         filename = os.path.basename(file)
#         match = version_pattern.search(filename)
#         if match:
#             version_num = int(match.group(2))
#             if version_num > highest_version:
#                 highest_version = version_num
#                 latest_file = file
# 
#     return latest_file
# 
# 
# ## Version
# def increment_version(dirname, fname, ext, version_prefix="_v"):
#     # Create a regex pattern to match the version number in the filename
#     version_pattern = re.compile(
#         rf"({re.escape(fname)}{re.escape(version_prefix)})(\d+)({re.escape(ext)})$"
#     )
# 
#     # Construct the glob pattern to find all files that match the pattern
#     glob_pattern = os.path.join(dirname, f"{fname}{version_prefix}*{ext}")
# 
#     # Use glob to find all files that match the pattern
#     files = glob(glob_pattern)
# 
#     # Initialize the highest version number
#     highest_version = 0
#     base, suffix = None, None
# 
#     # Loop through the files to find the highest version number
#     for file in files:
#         filename = os.path.basename(file)
#         match = version_pattern.search(filename)
#         if match:
#             base, version_str, suffix = match.groups()
#             version_num = int(version_str)
#             if version_num > highest_version:
#                 highest_version = version_num
# 
#     # If no versioned files were found, use the provided filename and extension
#     if base is None or suffix is None:
#         base = f"{fname}{version_prefix}"
#         suffix = ext
#         highest_version = 0  # No previous versions
# 
#     # Increment the highest version number
#     next_version_number = highest_version + 1
# 
#     # Format the next version number with the same number of digits as the original
#     next_version_str = f"{base}{next_version_number:03d}{suffix}"
# 
#     # Combine the directory and new filename to create the full path
#     next_filepath = os.path.join(dirname, next_version_str)
# 
#     return next_filepath
# 
# 
# if __name__ == "__main__":
#     import mngs
# 
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
# 
#     # (YOUR AWESOME CODE)
# 
#     # Close
#     mngs.gen.close(CONFIG)
# 
# # EOF
# 
# """
# /ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/path/_version.py
# """
# 
# 
# # EOF

#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.path._version import *

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
