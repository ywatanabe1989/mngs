# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/decorators/_cache_disk.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-07 06:08:45 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_cache_disk.py
# 
# import functools
# import os
# 
# from joblib import Memory as _Memory
# 
# 
# def cache_disk(func):
#     """Disk caching decorator that uses joblib.Memory.
# 
#     Usage:
#         @cache_disk
#         def expensive_function(x):
#             return x ** 2
#     """
#     mngs_dir = os.getenv("MNGS_DIR", "~/.cache/mngs/")
#     cache_dir = mngs_dir + "cache/"
#     memory = _Memory(cache_dir, verbose=0)
# 
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         cached_func = memory.cache(func)
#         return cached_func(*args, **kwargs)
# 
#     return wrapper
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

from mngs.decorators._cache_disk import *

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
