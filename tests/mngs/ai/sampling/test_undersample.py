# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-24 10:13:17 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/ai/sampling/undersample.py
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/ai/sampling/undersample.py"
# 
# from ...types import ArrayLike
# from imblearn.under_sampling import RandomUnderSampler
# 
# def undersample(X: ArrayLike, y: ArrayLike, random_state: int = 42) -> Tuple[ArrayLike, ArrayLike]:
#     """Undersample data preserving input type.
# 
#     Args:
#         X: Features array-like of shape (n_samples, n_features)
#         y: Labels array-like of shape (n_samples,)
#     Returns:
#         Resampled X, y of same type as input
#     """
#     rus = RandomUnderSampler(random_state=random_state)
#     X_resampled, y_resampled = rus.fit_resample(X, y)
#     return X_resampled, y_resampled
# 
# 
# # EOF

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

from mngs..ai.sampling.undersample import *

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
