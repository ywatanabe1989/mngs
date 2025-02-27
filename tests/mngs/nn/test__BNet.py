# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2023-05-15 16:44:27 (ywatanabe)"
# 
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
# import mngs
# import numpy as np
# import mngs
# 
# 
# class BHead(nn.Module):
#     def __init__(self, n_chs_in, n_chs_out):
#         super().__init__()
#         self.sa = mngs.nn.SpatialAttention(n_chs_in)
#         self.conv11 = nn.Conv1d(
#             in_channels=n_chs_in, out_channels=n_chs_out, kernel_size=1
#         )
# 
#     def forward(self, x):
#         x = self.sa(x)
#         x = self.conv11(x)
#         return x
# 
# 
# class BNet(nn.Module):
#     def __init__(self, BNet_config, MNet_config):
#         super().__init__()
#         self.dummy_param = nn.Parameter(torch.empty(0))
#         N_VIRTUAL_CHS = 32
# 
#         self.sc = mngs.nn.SwapChannels()
#         self.dc = mngs.nn.DropoutChannels(dropout=0.01)        
#         self.fgc = mngs.nn.FreqGainChanger(
#             BNet_config["n_bands"], BNet_config["SAMP_RATE"]
#         )
#         self.heads = nn.ModuleList(
#             [
#                 BHead(n_ch, N_VIRTUAL_CHS).to(self.dummy_param.device)
#                 for n_ch in BNet_config["n_chs"]
#             ]
#         )
# 
#         self.cgcs = [mngs.nn.ChannelGainChanger(n_ch) for n_ch in BNet_config["n_chs"]]
#         # self.cgc = mngs.nn.ChannelGainChanger(N_VIRTUAL_CHS)        
# 
#         MNet_config["n_chs"] = N_VIRTUAL_CHS  # BNet_config["n_chs"] # override
#         self.MNet = mngs.nn.MNet_1000(MNet_config)
# 
#         self.fcs = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     # nn.Linear(N_FC_IN, config["n_fc1"]),
#                     nn.Mish(),
#                     nn.Dropout(BNet_config["d_ratio1"]),
#                     nn.Linear(BNet_config["n_fc1"], BNet_config["n_fc2"]),
#                     nn.Mish(),
#                     nn.Dropout(BNet_config["d_ratio2"]),
#                     nn.Linear(BNet_config["n_fc2"], BNet_config["n_classes"][i_head]),
#                 )
#                 for i_head, _ in enumerate(range(len(BNet_config["n_chs"])))
#             ]
#         )
# 
#     @staticmethod
#     def _znorm_along_the_last_dim(x):
#         return (x - x.mean(dim=-1, keepdims=True)) / x.std(dim=-1, keepdims=True)
# 
#     def forward(self, x, i_head):
#         x = self._znorm_along_the_last_dim(x)        
#         # x = self.sc(x)
#         x = self.dc(x)        
#         x = self.fgc(x)
#         x = self.cgcs[i_head](x)
#         x = self.heads[i_head](x)
#         import ipdb; ipdb.set_trace()
#         # x = self.cgc(x)
#         x = self.MNet.forward_bb(x)
#         x = self.fcs[i_head](x)
#         return x
# 
# 
# # BNet_config = {
# #     "n_chs": 32,
# #     "n_bands": 6,
# #     "SAMP_RATE": 1000,
# # }
# BNet_config = {
#     "n_bands": 6,
#     "SAMP_RATE": 250,
#     # "n_chs": 270,
#     "n_fc1": 1024,
#     "d_ratio1": 0.85,
#     "n_fc2": 256,
#     "d_ratio2": 0.85,
# }
# 
# 
# if __name__ == "__main__":
#     ## Demo data
#     # MEG
#     BS, N_CHS, SEQ_LEN = 16, 160, 1000
#     x_MEG = torch.rand(BS, N_CHS, SEQ_LEN).cuda()
#     # EEG
#     BS, N_CHS, SEQ_LEN = 16, 19, 1000
#     x_EEG = torch.rand(BS, N_CHS, SEQ_LEN).cuda()
# 
#     # model = MNetBackBorn(mngs.nn.MNet_config).cuda()
#     # model(x_MEG)
#     # Model
#     BNet_config["n_chs"] = [160, 19]
#     BNet_config["n_classes"] = [2, 4]    
#     model = BNet(BNet_config, mngs.nn.MNet_config).cuda()
# 
#     # MEG
#     y = model(x_MEG, 0)
#     y = model(x_EEG, 1)    
# 
#     # # EEG
#     # y = model(x_EEG)
# 
#     y.sum().backward()

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

from ...src.mngs..nn._BNet import *

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
