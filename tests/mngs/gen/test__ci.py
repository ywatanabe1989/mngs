# Add your tests here

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/gen/_ci.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-04 06:55:56 (ywatanabe)"
# # /home/ywatanabe/proj/mngs/src/mngs/gen/_ci.py
# 
# 
# import numpy as np
# 
# 
# def ci(xx, axis=None):
#     indi = ~np.isnan(xx)
#     return 1.96 * (xx[indi]).std(axis=axis) / np.sqrt(indi.sum())

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/gen/_ci.py
# --------------------------------------------------------------------------------
