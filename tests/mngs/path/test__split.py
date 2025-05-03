# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/path/_split.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 16:18:06 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/path/_split.py
# 
# import os
# 
# def split(fpath):
#     """Split a file path to (1) the directory path, (2) the file name, and (3) the file extention
#     Example:
#         dirname, fname, ext = split('../data/01/day1/split_octave/2kHz_mat/tt8-2.mat')
#         print(dirname) # '../data/01/day1/split_octave/2kHz_mat/'
#         print(fname) # 'tt8-2'
#         print(ext) # '.mat'
#     """
#     dirname = os.path.dirname(fpath) + "/"
#     base = os.path.basename(fpath)
#     fname, ext = os.path.splitext(base)
#     return dirname, fname, ext
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/path/_split.py
# --------------------------------------------------------------------------------
