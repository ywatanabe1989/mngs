# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 04:00:36 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/gen/_color_text.py
# 
# def color_text(text, c="green"):
#     """Apply ANSI color codes to text.
# 
#     Parameters
#     ----------
#     text : str
#         The text to be colored.
#     c : str, optional
#         The color to apply. Available colors are 'red', 'green', 'yellow',
#         'blue', 'magenta', 'cyan', 'white', and 'grey' (default is "green").
# 
#     Returns
#     -------
#     str
#         The input text with ANSI color codes applied.
# 
#     Example
#     -------
#     >>> print(color_text("Hello, World!", "blue"))
#     # This will print "Hello, World!" in blue text
#     """
#     ANSI_COLORS = {
#         "red": "\033[91m",
#         "green": "\033[92m",
#         "yellow": "\033[93m",
#         "blue": "\033[94m",
#         "magenta": "\033[95m",
#         "cyan": "\033[96m",
#         "white": "\033[97m",
#         "grey": "\033[90m",
#         "gray": "\033[90m",
#         "reset": "\033[0m",
#     }
#     ANSI_COLORS["tra"] = ANSI_COLORS["white"]
#     ANSI_COLORS["val"] = ANSI_COLORS["green"]
#     ANSI_COLORS["tes"] = ANSI_COLORS["red"]
# 
#     start_code = ANSI_COLORS.get(c, ANSI_COLORS["reset"])
#     end_code = ANSI_COLORS["reset"]
#     return f"{start_code}{text}{end_code}"
# 
# 
# ct = color_text
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

from src.mngs.str._color_text import *

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
