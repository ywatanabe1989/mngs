# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/gen/_ci.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-04 06:55:56 (ywatanabe)"
# # /home/ywatanabe/proj/mngs/src/mngs/gen/_ci.py
# 
# 
# import numpy as np
# 
# 
# def ci(xx, axis=None):
#     indi = ~np.isnan(xx)
#     return 1.96 * (xx[indi]).std(axis=axis) / np.sqrt(indi.sum())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.gen._ci import *

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
