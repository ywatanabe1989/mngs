#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 12:01:37 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/ai/metrics/test__bACC.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/ai/metrics/test__bACC.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/ai/metrics/_bACC.py
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

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/ai/metrics/_bACC.py
# --------------------------------------------------------------------------------
