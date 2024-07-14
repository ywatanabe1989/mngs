#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-06 07:39:50 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/linalg/_distance.py


import scipy.spatial.distance as _distance
from mngs.gen import wrap


@wrap
def euclidean_distance(*args, **kwargs):
    return _distance.euclidean(*args, **kwargs)


@wrap
def cdist(*args, **kwargs):
    return _distance.cdist(*args, **kwargs)


# Optionally, manually copy the original docstring
euclidean_distance.__doc__ = _distance.euclidean.__doc__
cdist.__doc__ = _distance.cdist.__doc__
