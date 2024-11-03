# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-11 21:50:09 (ywatanabe)"
# 
# import torch
# import torch.nn as nn
# 
# 
# class PSD(nn.Module):
#     def __init__(self, sample_rate, prob=False, dim=-1):
#         super(PSD, self).__init__()
#         self.sample_rate = sample_rate
#         self.dim = dim
#         self.prob = prob
# 
#     def forward(self, signal):
# 
#         is_complex = signal.is_complex()
#         if is_complex:
#             signal_fft = torch.fft.fft(signal, dim=self.dim)
#             freqs = torch.fft.fftfreq(
#                 signal.size(self.dim), 1 / self.sample_rate
#             ).to(signal.device)
# 
#         else:
#             signal_fft = torch.fft.rfft(signal, dim=self.dim)
#             freqs = torch.fft.rfftfreq(
#                 signal.size(self.dim), 1 / self.sample_rate
#             ).to(signal.device)
# 
#         power_spectrum = torch.abs(signal_fft) ** 2
#         power_spectrum = power_spectrum / signal.size(self.dim)
# 
#         psd = power_spectrum * (1.0 / self.sample_rate)
# 
#         # To probability if specified
#         if self.prob:
#             psd /= psd.sum(dim=self.dim, keepdims=True)
# 
#         return psd, freqs

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

from src.mngs.nn._PSD import *

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
