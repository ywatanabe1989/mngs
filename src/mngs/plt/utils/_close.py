#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-29 20:41:30 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_close.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_close.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import matplotlib.pyplot as plt
import mngs.plt as mngs_plt


def close(obj):
    if isinstance(obj, matplotlib.figure.Figure):
        plt.close(obj)
    elif isinstance(obj, mngs_plt._subplots._FigWrapper.FigWrapper):
        plt.close(obj.figure)
    else:
        raise TypeError(
            f"Cannot close object of type {type(obj).__name__}. Expected FigWrapper or Figure object."
        )

# EOF