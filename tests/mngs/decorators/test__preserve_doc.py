# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/decorators/_preserve_doc.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:44:00 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_preserve_doc.py
# 
# from functools import wraps
# 
# def preserve_doc(loader_func):
#     """Wrap the loader functions to preserve their docstrings"""
#     @wraps(loader_func)
#     def wrapper(*args, **kwargs):
#         return loader_func(*args, **kwargs)
# 
#     return wrapper
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/decorators/_preserve_doc.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
