#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 19:45:54 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__mk_colorbar.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__mk_colorbar.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/_mk_colorbar.py
# --------------------------------------------------------------------------------
# import mngs
# import numpy as np
# import matplotlib
# # matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
#
#
# def mk_colorbar(start="white", end="blue"):
#     xx = np.linspace(0, 1, 256)
#
#     start = np.array(mngs.plt.colors.RGB_d[start])
#     end = np.array(mngs.plt.colors.RGB_d[end])
#     colors = (end-start)[:, np.newaxis]*xx
#
#     colors -= colors.min()
#     colors /= colors.max()
#
#     fig, ax = plt.subplots()
#     [ax.axvline(_xx, color=colors[:,i_xx]) for i_xx, _xx in enumerate(xx)]
#     ax.xaxis.set_ticks_position("none")
#     ax.yaxis.set_ticks_position("none")
#     ax.set_aspect(0.2)
#     return fig
#
#

import sys
from pathlib import Path

import mngs.plt._mk_colorbar as cbmod

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))


def test_mk_colorbar(monkeypatch):
    dummy_map = {"white": [255, 255, 255], "blue": [0, 0, 255]}
    monkeypatch.setattr(
        cbmod.mngs.plt.colors, "RGB_d", dummy_map, raising=False
    )
    fig = cbmod.mk_colorbar(start="white", end="blue")
    assert hasattr(fig, "axes")
    assert len(fig.axes) == 1


def test_mk_colorbar_outputs_figure(monkeypatch):
    dummy = {"white": [1, 1, 1], "blue": [0, 0, 1]}
    monkeypatch.setitem(cbmod.mngs.plt.colors.__dict__, "RGB_d", dummy)
    fig = cbmod.mk_colorbar(start="white", end="blue")
    assert hasattr(fig, "axes")
    assert len(fig.axes) == 1

# EOF