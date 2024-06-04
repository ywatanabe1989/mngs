#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-04 08:33:07 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/linalg/_distance.py


import scipy.spatial.distance as _distance
from mngs.gen import wrap


@wrap
def euclidean_distance(*args, **kwargs):
    return _distance.euclidean(*args, **kwargs)


@wrap
def cdist(*args, **kwargs):
    return _distance.cdist(*args, **kwargs)
