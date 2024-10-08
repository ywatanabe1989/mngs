#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-19 08:51:15 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/general/_cache.py


import time
from functools import lru_cache

cache = lru_cache(maxsize=None)
