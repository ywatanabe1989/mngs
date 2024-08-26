#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-22 11:40:12 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/io/_glob.py

import re
from natsort import natsorted
from glob import glob as _glob


def glob(expression):
    glob_pattern = re.sub(r"{[^}]*}", "*", expression)
    try:
        return natsorted(_glob(eval(glob_pattern)))
    except:
        return natsorted(_glob(glob_pattern))
