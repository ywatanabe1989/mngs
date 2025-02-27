# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Time-stamp: "2021-11-27 18:45:23 (ylab)"
# 
# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
# 
# 
# def mk_patches(colors, labels):
#     """
#     colors = ["red", "blue"]
#     labels = ["label_1", "label_2"]
#     ax.legend(handles=mngs.plt.mk_patches(colors, labels))
#     """
# 
#     patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
#     return patches

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

from ...src.mngs..plt._mk_patches import *

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
