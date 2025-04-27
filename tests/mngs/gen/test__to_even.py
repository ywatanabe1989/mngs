# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/_to_even.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-25 23:40:12 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/gen/_to_even.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/gen/_to_even.py"
# 
# def to_even(n):
#     """Convert a number to the nearest even number less than or equal to itself.
# 
#     Parameters
#     ----------
#     n : int or float
#         The input number to be converted.
# 
#     Returns
#     -------
#     int
#         The nearest even number less than or equal to the input.
# 
#     Example
#     -------
#     >>> to_even(5)
#     4
#     >>> to_even(6)
#     6
#     >>> to_even(3.7)
#     2
#     """
#     return int(n) - (int(n) % 2)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/_to_even.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
