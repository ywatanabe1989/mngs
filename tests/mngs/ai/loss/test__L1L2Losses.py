# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-07 18:53:03 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/ai/loss/_L1L2Losses.py
# 
# import torch
# 
# 
# def l1(model, lambda_l1=0.01):
#     lambda_l1 = torch.tensor(lambda_l1)
#     l1 = torch.tensor(0.0).cuda()
#     for param in model.parameters(): # fixme; is this OK?
#         l1 += torch.abs(param).sum()
#     return l1
# 
# 
# def l2(model, lambda_l2=0.01):
#     lambda_l2 = torch.tensor(lambda_l2)
#     l2 = torch.tensor(0.0).cuda()
#     for param in model.parameters(): # fixme; is this OK?
#         l2 += torch.norm(param).sum()
#     return l2
# 
# def elastic(model, alpha=1.0, l1_ratio=0.5):
#     assert 0 <= l1_ratio <= 1
# 
#     L1 = l1(model)
#     L2 = l2(model)
# 
#     return alpha * (l1_ratio * L1 + (1 - l1_ratio) * L2)
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

from mngs..ai.loss._L1L2Losses import *

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
