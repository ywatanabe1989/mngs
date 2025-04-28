# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/io/_load_modules/_pickle.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:41:33 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_load_modules/_pickle.py
# 
# import pickle
# 
# 
# def _load_pickle(lpath, **kwargs):
#     """Load pickle file."""
#     if not lpath.endswith(".pkl"):
#         raise ValueError("File must have .pkl extension")
#     with open(lpath, "rb") as f:
#         return pickle.load(f, **kwargs)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/io/_load_modules/_pickle.py
# --------------------------------------------------------------------------------
