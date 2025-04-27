# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dsp/reference.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "ywatanabe (2024-11-02 22:48:44)"
# # File: ./mngs_repo/src/mngs/dsp/reference.py
# 
# import torch as _torch
# from ..decorators import torch_fn as _torch_fn
# 
# 
# @_torch_fn
# def common_average(x, dim=-2):
#     re_referenced = (x - x.mean(dim=dim, keepdims=True)) / x.std(
#         dim=dim, keepdims=True
#     )
#     assert x.shape == re_referenced.shape
#     return re_referenced
# 
# @_torch_fn
# def random(x, dim=-2):
#     idx_all = [slice(None)] * x.ndim
#     idx_rand_dim = _torch.randperm(x.shape[dim])
#     idx_all[dim] = idx_rand_dim
#     y = x[idx_all]
#     re_referenced = x - y
#     assert x.shape == re_referenced.shape
#     return re_referenced
# 
# @_torch_fn
# def take_reference(x, tgt_indi, dim=-2):
#     idx_all = [slice(None)] * x.ndim
#     idx_all[dim] = tgt_indi
#     re_referenced = x - x[tgt_indi]
#     assert x.shape == re_referenced.shape
#     return re_referenced
# 
# if __name__ == "__main__":
#     import mngs
# 
#     x, f, t = mngs.dsp.demo_sig()
#     y = common_average(x)
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dsp/reference.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
