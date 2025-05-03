#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 12:00:45 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/ai/act/test__define.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/ai/act/test__define.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/ai/act/_define.py
# --------------------------------------------------------------------------------
# import torch.nn as nn
# 
# def define(act_str):
#     acts_dict = {
#         "relu": nn.ReLU(),
#         "swish": nn.SiLU(),
#         "mish": nn.Mish(),
#         "lrelu": nn.LeakyReLU(0.1),
#         }
#     return acts_dict[act_str]

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/ai/act/_define.py
# --------------------------------------------------------------------------------
