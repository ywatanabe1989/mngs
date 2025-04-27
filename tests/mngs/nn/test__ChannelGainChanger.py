# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/nn/_ChannelGainChanger.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2023-04-23 11:02:45 (ywatanabe)"
# 
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
# import mngs
# import numpy as np
# 
# 
# class ChannelGainChanger(nn.Module):
#     def __init__(
#         self,
#         n_chs,
#     ):
#         super().__init__()
#         self.n_chs = n_chs
# 
#     def forward(self, x):
#         """x: [batch_size, n_chs, seq_len]"""
#         if self.training:
#             ch_gains = (
#                 torch.rand(self.n_chs).unsqueeze(0).unsqueeze(-1).to(x.device) + 0.5
#             )
#             ch_gains = F.softmax(ch_gains, dim=1)
#             x *= ch_gains
# 
#         return x
# 
# 
# if __name__ == "__main__":
#     ## Demo data
#     bs, n_chs, seq_len = 16, 360, 1000
#     x = torch.rand(bs, n_chs, seq_len)
# 
#     cgc = ChGainChanger(n_chs)
#     print(cgc(x).shape)  # [16, 19, 1000]
# 
#     # sb = SubjectBlock(n_chs=n_chs)
#     # print(sb(x, s).shape) # [16, 270, 1000]
# 
#     # summary(sb, x, s)

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

from mngs.nn._ChannelGainChanger import *

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
