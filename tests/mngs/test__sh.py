# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-03 02:23:16 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/_sh.py
# 
# import subprocess
# import mngs
# 
# def sh(command_str, verbose=True):
#     """
#     Executes a shell command from Python.
# 
#     Parameters:
#     - command_str (str): The command string to execute.
# 
#     Returns:
#     - output (str): The standard output from the executed command.
#     """
#     if verbose:
#         print(mngs.gen.color_text(f"{command_str}", "yellow"))
# 
#     process = subprocess.Popen(
#         command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
#     )
#     output, error = process.communicate()
#     if process.returncode == 0:
#         out = output.decode("utf-8").strip()
#     else:
#         out = error.decode("utf-8").strip()
# 
#     if verbose:
#         print(out)
# 
#     return out
# 
# 
# if __name__ == "__main__":
#     import sys
# 
#     import matplotlib.pyplot as plt
#     import mngs
# 
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False
#     )
#     sh("ls")
#     mngs.gen.close(CONFIG, verbose=False, notify=False)
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

from mngs.._sh import *

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
