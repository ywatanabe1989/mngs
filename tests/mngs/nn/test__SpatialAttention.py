# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/nn/_SpatialAttention.py
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/nn/_SpatialAttention.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
