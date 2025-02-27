# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-05 07:35:30 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/pd/_ignore_.py
# 
# def ignore_SettingWithCopyWarning():
#     import warnings
#     try:
#         from pandas.errors import SettingWithCopyWarning
#     except:
#         from pandas.core.common import SettingWithCopyWarning
#     warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
#     # return SettingWithCopyWarning
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

from ...src.mngs..pd._ignore_SettingWithCopyWarning import *

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
