# src from here --------------------------------------------------------------------------------
# import pandas as pd
# import torch
# import numpy as np
# 
# def _format_a_sample_for_sktime(x):
#     """
#     x.shape: (n_chs, seq_len)
#     """
#     dims = pd.Series(
#         [pd.Series(x[d], name=f"dim_{d}") for d in range(len(x))],
#         index=[f"dim_{i}" for i in np.arange(len(x))]
#     )
#     return dims
# 
# def format_samples_for_sktime(X):
#     """
#     X.shape: (n_samples, n_chs, seq_len)
#     """
#     if torch.is_tensor(X):
#         X = X.numpy() # (64, 160, 1024)
# 
#         X = X.astype(np.float64)
# 
#     return pd.DataFrame(
#         [_format_a_sample_for_sktime(X[i]) for i in range(len(X))]
#         )

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

from ...src.mngs..ai.utils._format_samples_for_sktime import *

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
