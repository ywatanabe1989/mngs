# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/ai/utils/_check_params.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Time-stamp: "2024-02-17 12:38:40 (ywatanabe)"
# 
# from pprint import pprint as _pprint
# from time import sleep
# 
# # def get_params(model, tgt_name=None, sleep_sec=2, show=False):
# 
# #     name_shape_dict = {}
# #     for name, param in model.named_parameters():
# #         learnable = "Learnable" if param.requires_grad else "Freezed"
# 
# #         if (tgt_name is not None) & (name == tgt_name):
# #             return param
# #         if tgt_name is None:
# #             # print(f"\n{param}\n{param.shape}\nname: {name}\n")
# #             if show is True:
# #                 print(
# #                     f"\n{param}: {param.shape}\nname: {name}\nStatus: {learnable}\n"
# #                 )
# #                 sleep(sleep_sec)
# #             name_shape_dict[name] = list(param.shape)
# 
# #     if tgt_name is None:
# #         print()
# #         _pprint(name_shape_dict)
# #         print()
# 
# 
# def check_params(model, tgt_name=None, show=False):
# 
#     out_dict = {}
# 
#     for name, param in model.named_parameters():
#         learnable = "Learnable" if param.requires_grad else "Freezed"
# 
#         if tgt_name is None:
#             out_dict[name] = (param.shape, learnable)
# 
#         elif (tgt_name is not None) & (name == tgt_name):
#             out_dict[name] = (param.shape, learnable)
# 
#         elif (tgt_name is not None) & (name != tgt_name):
#             continue
# 
#     if show:
#         for k, v in out_dict.items():
#             print(f"\n{k}\n{v}")
# 
#     return out_dict

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

from mngs.ai.utils._check_params import *

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
