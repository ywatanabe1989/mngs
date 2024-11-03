# src from here --------------------------------------------------------------------------------
# import torch
# import warnings
# 
# def verify_n_gpus(n_gpus):
#     if torch.cuda.device_count() < n_gpus:
#         warnings.warn(
#             f"N_GPUS ({n_gpus}) is larger "
#             f"than n_gpus torch can acesses (= {torch.cuda.device_count()})"
#             f"Please check $CUDA_VISIBLE_DEVICES and your setting in this script.",
#             UserWarning,
#         )
#         return torch.cuda.device_count()
# 
#     else:
#         return n_gpus

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

from src.mngs.ai.utils._verify_n_gpus import *

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
