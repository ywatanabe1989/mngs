#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 16:08:22 (ywatanabe)"
# File: ./mngs_repo/src/mngs/typing/_ArrayLike.py

from typing import List, Tuple, Union

import numpy as _np
import pandas as _pd
import torch as _torch
import xarray as _xr

ArrayLike = Union[
    List,
    Tuple,
    _np.ndarray,
    _pd.Series,
    _pd.DataFrame,
    _xr.DataArray,
    _torch.Tensor,
]


# EOF
