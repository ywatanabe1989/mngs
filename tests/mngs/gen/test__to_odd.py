# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/_to_odd.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-25 23:40:22 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/gen/_to_odd.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/gen/_to_odd.py"
# 
# def to_odd(n):
#     """Convert a number to the nearest odd number less than or equal to itself.
# 
#     Parameters
#     ----------
#     n : int or float
#         The input number to be converted.
# 
#     Returns
#     -------
#     int
#         The nearest odd number less than or equal to the input.
# 
#     Example
#     -------
#     >>> to_odd(6)
#     5
#     >>> to_odd(7)
#     7
#     >>> to_odd(5.8)
#     5
#     """
#     return int(n) - ((int(n) + 1) % 2)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/_to_odd.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
