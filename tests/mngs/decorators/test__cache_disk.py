# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/decorators/_cache_disk.py
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

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/decorators/_cache_disk.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
