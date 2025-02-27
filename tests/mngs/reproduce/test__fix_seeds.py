# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 14:28:12 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/reproduce/_fix_seeds.py
# 
# def fix_seeds(
#     os=None, random=None, np=None, torch=None, tf=None, seed=42, verbose=True
# ):
#     os_str = "os" if os is not None else ""
#     random_str = "random" if random is not None else ""
#     np_str = "np" if np is not None else ""
#     torch_str = "torch" if torch is not None else ""
#     tf_str = "tf" if tf is not None else ""
# 
#     # https://github.com/lucidrains/vit-pytorch/blob/main/examples/cats_and_dogs.ipynb
#     if os is not None:
#         import os
# 
#         os.environ["PYTHONHASHSEED"] = str(seed)
# 
#     if random is not None:
#         random.seed(seed)
# 
#     if np is not None:
#         np.random.seed(seed)
# 
#     if torch is not None:
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         # torch.use_deterministic_algorithms(True)
# 
#     if tf is not None:
#         tf.random.set_seed(seed)
# 
#     if verbose:
#         print(f"\n{'-'*40}\n")
#         print(
#             f"Random seeds of the following packages have been fixed as {seed}"
#         )
#         print(os_str, random_str, np_str, torch_str, tf_str)
#         print(f"\n{'-'*40}\n")
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

from mngs..reproduce._fix_seeds import *

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
