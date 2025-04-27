# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/dict/_pop_keys.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 12:40:04 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/dict/_pop_keys.py
# 
# import numpy as np
# 
# 
# def pop_keys(keys_list, keys_to_pop):
#     """Remove specified keys from a list of keys.
# 
#     Parameters
#     ----------
#     keys_list : list
#         The original list of keys.
#     keys_to_pop : list
#         The list of keys to remove from keys_list.
# 
#     Returns
#     -------
#     list
#         A new list with the specified keys removed.
# 
#     Example
#     -------
#     >>> keys_list = ['a', 'b', 'c', 'd', 'e', 'bde']
#     >>> keys_to_pop = ['b', 'd']
#     >>> pop_keys(keys_list, keys_to_pop)
#     ['a', 'c', 'e', 'bde']
#     """
#     indi_to_remain = [k not in keys_to_pop for k in keys_list]
#     keys_remainded_list = list(np.array(keys_list)[list(indi_to_remain)])
#     return keys_remainded_list
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

from mngs.dict._pop_keys import *

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
