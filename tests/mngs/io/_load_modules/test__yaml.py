# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:41:37 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_load_modules/_yaml.py
# 
# import yaml
# 
# 
# def _load_yaml(lpath, **kwargs):
#     """Load YAML file with optional key lowercasing."""
#     if not lpath.endswith((".yaml", ".yml")):
#         raise ValueError("File must have .yaml or .yml extension")
# 
#     lower = kwargs.pop("lower", False)
#     with open(lpath) as f:
#         obj = yaml.safe_load(f, **kwargs)
# 
#     if lower:
#         obj = {k.lower(): v for k, v in obj.items()}
#     return obj
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

from mngs..io._load_modules._yaml import *

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
