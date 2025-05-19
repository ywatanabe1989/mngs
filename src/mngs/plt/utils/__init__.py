#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-02 10:54:27 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/utils/__init__.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/utils/__init__.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from ._calc_bacc_from_conf_mat import calc_bacc_from_conf_mat
from ._calc_nice_ticks import calc_nice_ticks
from ._configure_mpl import configure_mpl
from ._histogram_utils import HistogramBinManager, histogram_bin_manager
from ._im2grid import im2grid
from ._is_valid_axis import is_valid_axis, assert_valid_axis
from ._mk_colorbar import mk_colorbar
from ._mk_patches import mk_patches

# EOF