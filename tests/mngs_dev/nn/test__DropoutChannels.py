# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/nn/_DropoutChannels.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2023-05-04 21:50:22 (ywatanabe)"
# 
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
# import mngs
# import numpy as np
# import random
# 
# 
# class DropoutChannels(nn.Module):
#     def __init__(
#         self,
#         dropout=0.5
#     ):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
# 
#     def forward(self, x):
#         """x: [batch_size, n_chs, seq_len]"""
#         if self.training:
#             orig_chs = torch.arange(x.shape[1])
#             
#             indi_orig = self.dropout(torch.ones(x.shape[1])).bool()
#             chs_to_shuffle = orig_chs[~indi_orig]
# 
#             x[:, chs_to_shuffle] = torch.randn(x[:, chs_to_shuffle].shape).to(x.device)
# 
#             # rand_chs = random.sample(list(np.array(chs_to_shuffle)), len(chs_to_shuffle))
# 
#             # swapped_chs = orig_chs.clone()
#             # swapped_chs[~indi_orig] = torch.LongTensor(rand_chs)
# 
#             # x = x[:, swapped_chs.long(), :]
# 
#         return x
# 
# 
# if __name__ == "__main__":
#     ## Demo data
#     bs, n_chs, seq_len = 16, 360, 1000
#     x = torch.rand(bs, n_chs, seq_len)
# 
#     dc = DropoutChannels(dropout=0.1)
#     print(dc(x).shape)  # [16, 19, 1000]
# 
#     # sb = SubjectBlock(n_chs=n_chs)
#     # print(sb(x, s).shape) # [16, 270, 1000]
# 
#     # summary(sb, x, s)

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

from mngs.nn._DropoutChannels import *

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
