# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dsp/utils/_ensure_even_len.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-10 11:59:49 (ywatanabe)"
# 
# 
# def ensure_even_len(x):
#     if x.shape[-1] % 2 == 0:
#         return x
#     else:
#         return x[..., :-1]

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dsp/utils/_ensure_even_len.py
# --------------------------------------------------------------------------------
