# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2023-04-23 09:45:28 (ywatanabe)"
# 
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
# import mngs
# import numpy as np
# 
# class SpatialAttention(nn.Module):
#     def __init__(
#             self, n_chs_in
#     ):
#         super().__init__()
#         self.aap = nn.AdaptiveAvgPool1d(1)
#         self.conv11 = nn.Conv1d(in_channels=n_chs_in, out_channels=1, kernel_size=1)
# 
#     def forward(self, x):
#         """x: [batch_size, n_chs, seq_len]"""
#         x_orig = x
#         x = self.aap(x)
#         x = self.conv11(x)
# 
#         return x * x_orig

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

from src.mngs.nn._SpatialAttention import *

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
