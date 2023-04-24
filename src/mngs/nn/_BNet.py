#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-04-23 11:34:52 (ywatanabe)"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import mngs
import numpy as np
import mngs


class BHead(nn.Module):
    def __init__(self, n_chs_in, n_chs_out):
        super().__init__()
        self.sa = mngs.nn.SpatialAttention(n_chs_in)
        self.conv11 = nn.Conv1d(
            in_channels=n_chs_in, out_channels=n_chs_out, kernel_size=1
        )

    def forward(self, x):
        x = self.sa(x)
        x = self.conv11(x)
        return x


class BNet(nn.Module):
    def __init__(self, BNet_config, MNet_config):
        super().__init__()
        
        self.fgc = mngs.nn.FreqGainChanger(BNet_config["n_bands"], BNet_config["SAMP_RATE"])
        self.sc = mngs.nn.SwapChannels()
        
        self.N_CHS_MEG = 160
        self.N_CHS_EEG = 19

        self.MEG_head = BHead(self.N_CHS_MEG, BNet_config["n_chs"])
        self.EEG_head = BHead(self.N_CHS_EEG, BNet_config["n_chs"])

        self.cgc = mngs.nn.ChannelGainChanger(BNet_config["n_chs"])

        MNet_config["n_chs"] = BNet_config["n_chs"] # override        
        self.MNet = mngs.nn.MNet_1000(MNet_config)

    def get_head(self, x):
        if x.shape[1] == self.N_CHS_MEG:
            return self.MEG_head
        if x.shape[1] == self.N_CHS_EEG:
            return self.EEG_head

    def forward(self, x):
        x = self.sc(x)
        x = self.fgc(x)

        head = self.get_head(x)
        x = head(x)
        x = self.cgc(x)                
        x = self.MNet(x)
        return x

BNet_config = {"n_chs": 32,
               "n_bands": 10,
               "SAMP_RATE": 1000,
               }

if __name__ == "__main__":
    ## Demo data
    # MEG
    BS, N_CHS, SEQ_LEN = 16, 160, 1000
    x_MEG = torch.rand(BS, N_CHS, SEQ_LEN).cuda()
    # EEG    
    BS, N_CHS, SEQ_LEN = 16, 19, 1000
    x_EEG = torch.rand(BS, N_CHS, SEQ_LEN).cuda()

    # Model
    model = BNet(BNet_config, mngs.nn.MNet_config).cuda()

    # MEG
    y = model(x_MEG)
    
    # # EEG
    # y = model(x_EEG)

    y.sum().backward()
