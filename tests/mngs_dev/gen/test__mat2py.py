# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/gen/_mat2py.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-03 18:57:14 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/gen/_mat2py.py
# 
# """Helper script for loading .mat files into python.
# For .mat with multiple variables use mat2dict to get return dictionary with .mat variables.
# For .mat with 1 matrix use mat2npa to return np.array
# For .mat with 1 matrix use mat2npy to save np.array to .npy
# For multiple .mat files with 1 matrix use dir2npy to save 1 np.array of each .mat to .npy
# 
# 
# Examples:
# mat2py.mat2npa(fname = '/vol/ccnlab-scratch1/julber/chill_nn_regression/data/chill_wav_time_16kHz.mat', typ = np.float32)
# mat2py.dir2npa(dir = '/vol/ccnlab-scratch1/julber/phoneme_decoding/data/', typ = np.float32, regex = '*xdata')
# mat2py.dir2npa(dir = '/vol/ccnlab-scratch1/julber/phoneme_decoding/data/', typ = np.int32, regex = '*ylabels')
# 
# 
# September 07, 2017
# JB"""
# 
# import numpy as np
# import h5py
# from glob import glob as _glob
# import os
# from scipy.io import loadmat
# 
# 
# def mat2dict(fname):
#     """Function returns a dictionary with .mat variables"""
#     try:
#         D = h5py.File(fname)
#         d = {}
#         for key, value in D.items():
#             d[key] = value
#         d["__hdf__"] = True
#     except:
#         d = loadmat(fname)
#         d["__hdf__"] = False
#     return d
# 
# 
# def keys2npa(d, typ):
#     import pdb
# 
#     pdb.set_trace()
#     d2 = {}
#     for key in public_keys(d):
#         x = np.array(d[key], dtype=typ)
#         if d["__hdf__"]:
#             x = np.squeeze(np.swapaxes(x, 0, -1))
#         assert type(x.flatten()[0]) == typ
#         d2[key] = x.copy()
#     return d2
# 
# 
# def public_keys(d):
#     return [k for k in d.keys() if not k.startswith("_")]
# 
# 
# def mat2npa(fname, typ):
#     """Function returns np array from 1st entry in .mat file"""
#     import pdb
# 
#     pdb.set_trace()
#     d = keys2npa(mat2dict(fname), typ)
#     return d[d.keys()[0]]
# 
# 
# def save_npa(fname, x):
#     np.save(fname, x)
# 
# 
# def mat2npy(fname, typ):
#     """Function save np array from 1st entry in .mat file to .npy file"""
#     x = mat2npa(fname, typ)
#     save_npa(fname=fname.replace(".mat", ""), x=x)
# 
# 
# def dir2npy(dir, typ, regex="*"):
#     """Function saves np array from 1st entry in each regex + .mat file in dir"""
#     os.chdir(dir)
#     for fname in _glob(regex + ".mat"):
#         print("File " + fname + " to" + " .npa")
#         mat2npy(dir + fname, typ)
# 
# 
# # EOF

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.gen._mat2py import *

class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        raise NotImplementedError("Test not yet implemented")

    def test_edge_cases(self):
        # Edge case testing
        raise NotImplementedError("Test not yet implemented")

    def test_error_handling(self):
        # Error handling testing
        raise NotImplementedError("Test not yet implemented")

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
