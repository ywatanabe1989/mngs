# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 04:04:31 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/str/_squeeze_space.py
# 
# import re
# 
# def squeeze_spaces(string, pattern=" +", repl=" "):
#     """Replace multiple occurrences of a pattern in a string with a single replacement.
# 
#     Parameters
#     ----------
#     string : str
#         The input string to be processed.
#     pattern : str, optional
#         The regular expression pattern to match (default is " +", which matches one or more spaces).
#     repl : str or callable, optional
#         The replacement string or function (default is " ", a single space).
# 
#     Returns
#     -------
#     str
#         The processed string with pattern occurrences replaced.
# 
#     Example
#     -------
#     >>> squeeze_spaces("Hello   world")
#     'Hello world'
#     >>> squeeze_spaces("a---b--c-d", pattern="-+", repl="-")
#     'a-b-c-d'
#     """
#     return re.sub(pattern, repl, string)
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

from mngs.str._squeeze_space import *

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
