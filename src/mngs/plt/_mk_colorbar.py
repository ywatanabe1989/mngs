#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 21:18:36 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_mk_colorbar.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_mk_colorbar.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib.pyplot as plt
import mngs
import numpy as np


def mk_colorbar(start="white", end="blue"):
    xx = np.linspace(0, 1, 256)

    start = np.array(mngs.plt.colors.RGB_d[start])
    end = np.array(mngs.plt.colors.RGB_d[end])
    colors = (end - start)[:, np.newaxis] * xx

    colors -= colors.min()
    colors /= colors.max()

    fig, ax = plt.subplots()
    [ax.axvline(_xx, color=colors[:, i_xx]) for i_xx, _xx in enumerate(xx)]
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.set_aspect(0.2)
    return fig

# EOF