# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/ai/metrics/_bACC.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-02-26 16:32:42 (ywatanabe)"
# 
# import warnings
# 
# import numpy as np
# import torch
# from sklearn.metrics import balanced_accuracy_score
# 
# 
# def bACC(true_class, pred_class):
#     """
#     Calculates the balanced accuracy score between predicted and true class labels.
# 
#     Parameters:
#     - true_class (array-like or torch.Tensor): True class labels.
#     - pred_class (array-like or torch.Tensor): Predicted class labels.
# 
#     Returns:
#     - bACC (float): The balanced accuracy score rounded to three decimal places.
#     """
#     if isinstance(true_class, torch.Tensor):  # [REVISED]
#         true_class = true_class.detach().cpu().numpy()  # [REVISED]
#     if isinstance(pred_class, torch.Tensor):  # [REVISED]
#         pred_class = pred_class.detach().cpu().numpy()  # [REVISED]
# 
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         bACC_score = balanced_accuracy_score(
#             true_class.reshape(-1),  # [REVISED]
#             pred_class.reshape(-1),  # [REVISED]
#         )
#     return round(bACC_score, 3)  # [REVISED]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.ai.metrics._bACC import *

class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        raise NotImplementedError("Test not yet implemented")

    def test_edge_cases(self):
        # Edge case testing
        raise NotImplementedError("Test not yet implemented")

    def test_error_handling(self):
        # Error handling testing
        raise NotImplementedError("Test not yet implemented")

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
