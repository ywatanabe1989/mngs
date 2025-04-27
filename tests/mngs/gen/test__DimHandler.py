# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/gen/_DimHandler.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-05 00:39:26 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/gen/_DimHandler.py
# 
# """
# This script demonstrates DimHandler, which:
# 1) Keeps designated dimensions,
# 2) Permutes the kept dimensions to the last while maintaining their relative order,
# 3) Reshapes the remaining dimensions to the first, batch dimension,
# 4) (Performs calculations),
# 5) Restores the summarized dimensions to their original shapes.
# """
# 
# # Imports
# import sys
# 
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# 
# 
# # Functions
# class DimHandler:
#     """
#     A utility class for handling dimension manipulations on tensors or arrays, including reshaping and permuting dimensions.
# 
#     Attributes:
#         orig_shape (tuple): The original shape of the input tensor or array before any manipulation.
#         keepdims (list): The list of dimensions to be kept and moved to the end.
#         n_non_keepdims (list): The sizes of the dimensions not kept, used for reshaping back to the original shape.
#         n_keepdims (list): The sizes of the kept dimensions, used for reshaping.
# 
#     Example1:
#         import torch
# 
#         dh = DimHandler()
#         x = torch.rand(1, 2, 3, 4, 5, 6)  # Example tensor
#         print(x.shape)  # torch.Size([1, 2, 3, 4, 5, 6])
#         x = dh.fit(x, keepdims=[0, 2, 5])
#         print(x.shape)  # torch.Size([40, 1, 3, 6])
#         x = dh.unfit(x)
#         print(x.shape)  # torch.Size([2, 4, 5, 1, 3, 6])
# 
#     Example 2:
#         import torch
# 
#         dh = DimHandler()
#         x = torch.rand(1, 2, 3, 4, 5, 6)  # Example tensor
#         print(x.shape)  # torch.Size([1, 2, 3, 4, 5, 6])
#         x = dh.fit(x, keepdims=[0, 2, 5])
#         print(x.shape)  # torch.Size([40, 1, 3, 6])
#         y = x.mean(axis=-2) # calculation on the kept dims
#         print(y.shape) # torch.Size([40, 1, 6])
#         y = dh.unfit(y)
#         print(y.shape) # torch.Size([2, 4, 5, 1, 6])
#     """
# 
#     def __init__(self):
#         pass
#         # self.orig_shape = None
#         # self.keepdims = None
# 
#     def fit(self, x, keepdims=[]):
#         if isinstance(x, np.ndarray):
#             return self._fit_numpy(x, keepdims=keepdims)
#         elif isinstance(x, torch.Tensor):
#             return self._fit_torch(x, keepdims=keepdims)
# 
#     def _fit_numpy(self, x, keepdims=[]):
#         """
#         Reshapes the input NumPy array by flattening the dimensions not in `keepdims` and moving the kept dimensions to the end.
# 
#         Arguments:
#             x (numpy.ndarray): The input array to be reshaped.
#             keepdims (list of int): The indices of the dimensions to keep.
# 
#         Returns:
#             x_flattened (numpy.ndarray): The reshaped array with kept dimensions moved to the end.
#         """
#         assert len(keepdims) <= len(
#             x.shape
#         ), "keepdims cannot have more dimensions than the array itself."
# 
#         # Normalize negative indices to positive indices
#         total_dims = len(x.shape)
#         keepdims = [dim if dim >= 0 else total_dims + dim for dim in keepdims]
#         keepdims = sorted(set(keepdims))
# 
#         self.shape_fit = x.shape
# 
#         non_keepdims = [ii for ii in range(len(self.shape_fit)) if ii not in keepdims]
# 
#         self.n_non_keepdims = [self.shape_fit[nkd] for nkd in non_keepdims]
#         self.n_keepdims = [self.shape_fit[kd] for kd in keepdims]
# 
#         # Permute the array dimensions so that the non-kept dimensions come first
#         new_order = non_keepdims + keepdims
#         x_permuted = np.transpose(x, axes=new_order)
# 
#         # Flatten the non-kept dimensions
#         x_flattened = x_permuted.reshape(-1, *self.n_keepdims)
# 
#         return x_flattened
# 
#     def _fit_torch(self, x, keepdims=[]):
#         """
#         Reshapes the input tensor or array by flattening the dimensions not in `keepdims` and moving the kept dimensions to the end.
# 
#         Arguments:
#             x (torch.Tensor): The input tensor or array to be reshaped.
#             keepdims (list of int): The indices of the dimensions to keep.
# 
#         Returns:
#             x_flattend (torch.Tensor): The reshaped tensor or array with kept dimensions moved to the end.
# 
#         Note:
#             This method modifies the `orig_shape`, `keepdims`, `n_non_keepdims`, and `n_keepdims` attributes based on the input.
#         """
#         assert len(keepdims) <= len(
#             x.shape
#         ), "keepdims cannot have more dimensions than the tensor itself."
# 
#         keepdims = torch.tensor(keepdims).clone().detach().cpu().int()
#         # Normalize negative indices to positive indices
#         total_dims = len(x.shape)
#         keepdims = [dim if dim >= 0 else total_dims + dim for dim in keepdims]
#         keepdims = sorted(set(keepdims))
# 
#         self.shape_fit = x.shape
# 
#         non_keepdims = [
#             int(ii) for ii in torch.arange(len(self.shape_fit)) if ii not in keepdims
#         ]
# 
#         self.n_non_keepdims = [self.shape_fit[nkd] for nkd in non_keepdims]
#         self.n_keepdims = [self.shape_fit[kd] for kd in keepdims]
# 
#         x_permuted = x.permute(*non_keepdims, *keepdims)
#         x_flattend = x_permuted.reshape(-1, *self.n_keepdims)
# 
#         return x_flattend
# 
#     def unfit(self, y):
#         """
#         Restores the first dimension of reshaped tensor or array back to its original shape before the `fit` operation.
# 
#         Arguments:
#             y (torch.Tensor or numpy.array): The tensor or array to be restored to its original shape.
# 
#         Returns:
#             y_restored (torch.Tensor or numpy.array): The tensor or array restored to its original shape.
#         """
#         self.shape_unfit = y.shape
#         return y.reshape(*self.n_non_keepdims, *self.shape_unfit[1:])
# 
# 
# if __name__ == "__main__":
#     import mngs
# 
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
# 
#     # Example1:
#     mngs.gen.printc("Example 1")
#     dh = DimHandler()
#     x = torch.rand(1, 2, 3, 4, 5, 6)  # Example tensor
#     print(x.shape)  # torch.Size([1, 2, 3, 4, 5, 6])
#     x = dh.fit(x, keepdims=[0, 2, 5])
#     print(x.shape)  # torch.Size([40, 1, 3, 6])
#     x = dh.unfit(x)
#     print(x.shape)  # torch.Size([2, 4, 5, 1, 3, 6])
# 
#     # Example 2:
#     mngs.gen.printc("Example 2")
#     dh = DimHandler()
#     x = torch.rand(1, 2, 3, 4, 5, 6)  # Example tensor
#     print(x.shape)  # torch.Size([1, 2, 3, 4, 5, 6])
#     x = dh.fit(x, keepdims=[0, 2, 5])
#     print(x.shape)  # torch.Size([40, 1, 3, 6])
#     y = x.mean(axis=-2)  # calculation on the kept dims
#     print(y.shape)  # torch.Size([40, 1, 6])
#     y = dh.unfit(y)
#     print(y.shape)  # torch.Size([2, 4, 5, 1, 6])
# 
#     # Close
#     mngs.gen.close(CONFIG)
# 
# # EOF
# 
# """
# /ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/gen/_DimHandler.py
# """
# 
# 
# # EOF

#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.gen._DimHandler import *

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
