# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dsp/_misc.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-05 01:03:32 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/dsp/_misc.py
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-05 12:14:08 (ywatanabe)"
# 
# from ..decorators import torch_fn
# 
# @torch_fn
# def ensure_3d(x):
#     if x.ndim == 1:  # assumes (seq_len,)
#         x = x.unsqueeze(0).unsqueeze(0)
#     elif x.ndim == 2:  # assumes (batch_siize, seq_len)
#         x = x.unsqueeze(1)
#     return x
# 
# 
# # @torch_fn
# # def unbias(x, dim=-1, fn="mean"):
# #     if fn == "mean":
# #         return x - x.mean(dim=dim, keepdims=True)
# #     if fn == "min":
# #         return x - x.min(dim=dim, keepdims=True)[0]
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dsp/_misc.py
# --------------------------------------------------------------------------------
