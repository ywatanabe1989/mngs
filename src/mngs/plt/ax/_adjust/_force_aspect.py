#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 20:31:50 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_force_aspect.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/ax/_force_aspect.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib


def force_aspect(axis, aspect=1):
    assert isinstance(
        axis, matplotlib.axses._axses.Axses
    ), "First argument must be a matplotlib axisis"

    im = axis.get_images()

    extent = im[0].get_extent()

    axis.set_aspect(
        abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect
    )
    return axis

# EOF