# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/nn/_AxiswiseDropout.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-03-30 07:27:27 (ywatanabe)"
# 
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# 
# 
# class AxiswiseDropout(nn.Module):
#     def __init__(self, dropout_prob=0.5, dim=1):
#         super(AxiswiseDropout, self).__init__()
#         self.dropout_prob = dropout_prob
#         self.dim = dim
# 
#     def forward(self, x):
#         if self.training:
#             sizes = [s if i == self.dim else 1 for i, s in enumerate(x.size())]
#             dropout_mask = F.dropout(
#                 torch.ones(*sizes, device=x.device, dtype=x.dtype),
#                 self.dropout_prob,
#                 True,
#             )
# 
#             # Expand the mask to the size of the input tensor and apply it
#             return x * dropout_mask.expand_as(x)
#         return x

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

from mngs.nn._AxiswiseDropout import *

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
