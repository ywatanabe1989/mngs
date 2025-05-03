# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/io/_load_modules/_torch.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:41:34 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_load_modules/_torch.py
# 
# import torch
# 
# 
# def _load_torch(lpath, **kwargs):
#     """Load PyTorch model/checkpoint file."""
#     if not lpath.endswith((".pth", ".pt")):
#         raise ValueError("File must have .pth or .pt extension")
#     return torch.load(lpath, **kwargs)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/io/_load_modules/_torch.py
# --------------------------------------------------------------------------------
