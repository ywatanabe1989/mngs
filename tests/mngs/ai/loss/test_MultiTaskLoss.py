#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 12:01:28 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/ai/loss/test_MultiTaskLoss.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/ai/loss/test_MultiTaskLoss.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/loss/MultiTaskLoss.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-07 19:07:29 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/ai/loss/MultiTaskLoss.py
# 
# import numpy as np
# import torch
# import torch.nn as nn
# 
# from ...reproduce import fix_seeds
# 
# 
# class MultiTaskLoss(nn.Module):
#     """
#     # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
# 
#     Example:
#         are_regression = [False, False]
#         mtl = MultiTaskLoss(are_regression)
#         losses = [torch.rand(1, requires_grad=True) for _ in range(len(are_regression))]
#         loss = mtl(losses)
#         print(loss)
#         # [tensor([0.4215], grad_fn=<AddBackward0>), tensor([0.6190], grad_fn=<AddBackward0>)]
#     """
# 
#     def __init__(self, are_regression=[False, False], reduction="none"):
#         super().__init__()
#         fix_seeds(np=np, torch=torch, show=False)
#         n_tasks = len(are_regression)
# 
#         self.register_buffer("are_regression", torch.tensor(are_regression))
# 
#         # for the numercal stability, log(variables) are learned.
#         self.log_vars = torch.nn.Parameter(torch.zeros(n_tasks))
#         self.reduction = reduction
# 
#     def forward(self, losses):
#         vars = torch.exp(self.log_vars).type_as(losses[0])
#         stds = vars ** (1 / 2)
#         coeffs = 1 / ((self.are_regression + 1) * vars)
#         scaled_losses = [
#             coeffs[i] * losses[i] + torch.log(stds[i]) for i in range(len(losses))
#         ]
#         return scaled_losses
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/loss/MultiTaskLoss.py
# --------------------------------------------------------------------------------
