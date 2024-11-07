# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-03-31 08:11:45 (ywatanabe)"
# 
# import torch
# 
# 
# def apply_to(fn, x, dim):
#     """
#     Example:
#     x = torch.randn(2, 3, 4)
#     fn = sum
#     apply_to(fn, x, 1).shape # (2, 1, 4)
#     """
#     if dim != -1:
#         dims = list(range(x.dim()))
#         dims[-1], dims[dim] = dims[dim], dims[-1]
#         x = x.permute(*dims)
# 
#     # Flatten the tensor along the time dimension
#     shape = x.shape
#     x = x.reshape(-1, shape[-1])
# 
#     # Apply the function to each slice along the time dimension
#     applied = torch.stack([fn(x_i) for x_i in torch.unbind(x, dim=0)], dim=0)
# 
#     # Reshape the tensor to its original shape (with the time dimension at the end)
#     applied = applied.reshape(*shape[:-1], -1)
# 
#     # Permute back to the original dimension order if necessary
#     if dim != -1:
#         applied = applied.permute(*dims)
# 
#     return applied

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

from mngs.torch._apply_to import *

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
