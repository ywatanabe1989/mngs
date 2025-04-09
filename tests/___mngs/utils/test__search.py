# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/utils/_search.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 13:01:38 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/utils/_search.py
# 
# import numpy as np
# import re
# 
# def search(patterns, strings, only_perfect_match=False, as_bool=False, ensure_one=False):
#     """Search for patterns in strings using regular expressions.
# 
#     Parameters
#     ----------
#     patterns : str or list of str
#         The pattern(s) to search for. Can be a single string or a list of strings.
#     strings : str or list of str
#         The string(s) to search in. Can be a single string or a list of strings.
#     only_perfect_match : bool, optional
#         If True, only exact matches are considered (default is False).
#     as_bool : bool, optional
#         If True, return a boolean array instead of indices (default is False).
#     ensure_one : bool, optional
#         If True, ensures only one match is found (default is False).
# 
#     Returns
#     -------
#     tuple
#         A tuple containing two elements:
#         - If as_bool is False: (list of int, list of str)
#           The first element is a list of indices where matches were found.
#           The second element is a list of matched strings.
#         - If as_bool is True: (numpy.ndarray of bool, list of str)
#           The first element is a boolean array indicating matches.
#           The second element is a list of matched strings.
# 
#     Example
#     -------
#     >>> patterns = ['orange', 'banana']
#     >>> strings = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
#     >>> search(patterns, strings)
#     ([1, 4, 5], ['orange', 'banana', 'orange_juice'])
# 
#     >>> patterns = 'orange'
#     >>> strings = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
#     >>> search(patterns, strings)
#     ([1, 5], ['orange', 'orange_juice'])
#     """
# 
#     def to_list(string_or_pattern):
#         if isinstance(string_or_pattern, (np.ndarray, pd.Series, xr.DataArray)):
#             return string_or_pattern.tolist()
#         elif isinstance(string_or_pattern, abc.KeysView):
#             return list(string_or_pattern)
#         elif not isinstance(string_or_pattern, (list, tuple, pd.Index)):
#             return [string_or_pattern]
#         return string_or_pattern
# 
#     patterns = to_list(patterns)
#     strings = to_list(strings)
# 
#     indices_matched = []
#     for pattern in patterns:
#         for index_str, string in enumerate(strings):
#             if only_perfect_match:
#                 if pattern == string:
#                     indices_matched.append(index_str)
#             else:
#                 if re.search(pattern, string):
#                     indices_matched.append(index_str)
# 
#     indices_matched = natsorted(indices_matched)
#     keys_matched = list(np.array(strings)[indices_matched])
# 
#     if ensure_one:
#         assert len(indices_matched) == 1, "Expected exactly one match, but found {}".format(len(indices_matched))
# 
#     if as_bool:
#         bool_matched = np.zeros(len(strings), dtype=bool)
#         bool_matched[np.unique(indices_matched)] = True
#         return bool_matched, keys_matched
#     else:
#         return indices_matched, keys_matched
# 
# 
# # EOF

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

from mngs.utils._search import *

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
