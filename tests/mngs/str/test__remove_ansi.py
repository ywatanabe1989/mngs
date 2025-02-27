# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 01:21:34 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/str/_remove_ansi.py
# 
# import re
# 
# def remove_ansi(string):
#     """
#     Removes ANSI escape sequences from a given text chunk.
# 
#     Parameters:
#     - chunk (str): The text chunk to be cleaned.
# 
#     Returns:
#     - str: The cleaned text chunk.
#     """
#     ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
#     return ansi_escape.sub("", string)
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

from mngs..str._remove_ansi import *

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
