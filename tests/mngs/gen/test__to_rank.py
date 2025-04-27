# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/gen/_to_rank.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 13:05:47 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/gen/_to_rank.py
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-08-29 22:10:06 (ywatanabe)"
# # ./src/mngs/gen/data_processing/_to_rank.py
# 
# import torch
# 
# # from .._converters import
# from ..decorators import torch_fn
# 
# 
# @torch_fn
# def to_rank(tensor, method="average"):
#     sorted_tensor, indices = torch.sort(tensor)
#     ranks = torch.empty_like(tensor)
#     ranks[indices] = (
#         torch.arange(len(tensor), dtype=tensor.dtype, device=tensor.device) + 1
#     )
# 
#     if method == "average":
#         ranks = ranks.float()
#         ties = torch.nonzero(sorted_tensor[1:] == sorted_tensor[:-1])
#         for i in range(len(ties)):
#             start = ties[i]
#             end = start + 1
#             while (
#                 end < len(sorted_tensor) and sorted_tensor[end] == sorted_tensor[start]
#             ):
#                 end += 1
#             ranks[indices[start:end]] = ranks[indices[start:end]].mean()
# 
#     return ranks
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

from mngs.gen._to_rank import *

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
