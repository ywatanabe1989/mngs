# src from here --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-24 12:59:28 (ywatanabe)"
# # /home/ywatanabe/proj/mngs_repo/src/mngs/ai/_gen_ai/_PRICING.py
# 
# 
# from .PARAMS import MODELS
# 
# 
# def calc_cost(model, input_tokens, output_tokens):
#     import pandas as pd
#     MODELs = pd.DataFrame(MODELS)
#     indi = MODELS["name"] == model
#     costs = MODELS[["input_cost", "output_cost"]][indi]
#     cost = (
#         input_tokens * costs["input_cost"]
#         + output_tokens * costs["output_cost"]
#     ) / 1_000_000
#     return cost.iloc[0]

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

from src.mngs.ai/_gen_ai/_calc_cost.py import *

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
