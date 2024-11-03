# src from here --------------------------------------------------------------------------------
# # import torch.nn as nn
# import torch.optim as optim
# from .Ranger_Deep_Learning_Optimizer.ranger.ranger2020 import Ranger
# 
# 
# # def set_an_optim(models, optim_str, lr):
# def set(models, optim_str, lr):    
#     """Sets an optimizer to models"""
#     if not isinstance(models, list):
#         models = [models]
#     learnable_params = []
#     for m in models:
#         learnable_params += list(m.parameters())
#     # optim = mngs.ml.switch_optim(optim_str)
#     optim = get(optim_str)    
#     return optim(learnable_params, lr)
# 
# def get(optim_str):
#     optims_dict = {
#         "adam": optim.Adam,
#         "ranger": Ranger,
#         "rmsprop": optim.RMSprop,
#         "sgd": optim.SGD
#         }
#     return optims_dict[optim_str]

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

from src.mngs.ai.optim._get_set import *

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
