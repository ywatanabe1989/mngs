# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# 
# 
# from collections import Counter
# 
# import numpy as np
# 
# 
# def under_sample(y, replace=False):
#     """
#     Input:
#         Labels
#     Return:
#         Indices
# 
#     Example:
#         t = ['a', 'b', 'c', 'b', 'c', 'a', 'c']
#         print(under_sample(t))
#         # [5 0 1 3 4 6]
#         print(under_sample(t))
#         # [5 0 1 3 6 2]
#     """
# 
#     # find the minority and majority classes
#     class_counts = Counter(y)
#     # majority_class = max(class_counts, key=class_counts.get)
#     minority_class = min(class_counts, key=class_counts.get)
# 
#     # compute the number of sample to draw from the majority class using
#     # a negative binomial distribution
#     n_minority_class = class_counts[minority_class]
#     n_majority_resampled = n_minority_class
# 
#     # draw randomly with or without replacement
#     indices = np.hstack(
#         [
#             np.random.choice(
#                 np.flatnonzero(y == k),
#                 size=n_majority_resampled,
#                 replace=replace,
#             )
#             for k in class_counts.keys()
#         ]
#     )
# 
#     return indices
# 
# 
# if __name__ == "__main__":
#     t = np.array(["a", "b", "c", "b", "c", "a", "c"])
#     print(under_sample(t))

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

from mngs.ai.utils._under_sample import *

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
