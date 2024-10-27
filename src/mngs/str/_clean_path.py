#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-25 12:13:39 (ywatanabe)"
# /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/str/_clean_path.py

"""
1. Functionality:
   - (e.g., Executes XYZ operation)
2. Input:
   - (e.g., Required data for XYZ)
3. Output:
   - (e.g., Results of XYZ operation)
4. Prerequisites:
   - (e.g., Necessary dependencies for XYZ)

(Remove me: Please fill docstrings above, while keeping the bulette point style, and remove this instruction line)
"""

"""Imports"""
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import matplotlib.pyplot as plt
import mngs

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from natsort import natsorted
from glob import glob
from pprint import pprint
import warnings
import logging
from tqdm import tqdm
import xarray as xr
from mngs.typing import List, Tuple, Dict, Any, Union, Sequence, Literal, Optional, Iterable, ArrayLike

try:
    from scripts import utils
except:
    pass

"""Parameters"""
# CONFIG = mngs.gen.load_configs()

"""Functions & Classes"""
def clean(string):
    string = string.replace("/./", "")

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt, verbose=False, agg=True)
    main()
    mngs.gen.close(CONFIG, verbose=False, sys=sys, notify=False, message="")

# EOF
