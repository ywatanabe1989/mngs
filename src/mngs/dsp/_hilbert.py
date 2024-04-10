#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-10 18:05:53 (ywatanabe)"

"""
This script does XYZ.
"""

import sys

import matplotlib.pyplot as plt
import mngs
import torch
from mngs.general import torch_fn
from mngs.nn import Hilbert

# Config
CONFIG = mngs.gen.load_configs()

# Functions
@torch_fn
def hilbert(
    x,
    dim=-1,
):
    y = Hilbert(dim=dim)(x)
    return y[..., 0], y[..., 1]


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    # Parameters
    T_SEC = 1.0
    FS = 400

    # Demo signal
    xx, tt, fs = mngs.dsp.demo_sig(t_sec=T_SEC, fs=FS, sig_type="chirp")

    # Main
    pha, amp = hilbert(
        xx,
        dim=-1,
    )
    # (32, 19, 1280, 2)

    # Plots
    fig, axes = mngs.plt.subplots(nrows=2, sharex=True)

    axes[0].plot(tt, xx[0, 0], label="orig")
    axes[0].plot(tt, amp[0, 0], label="amp")
    axes[0].legend()
    # axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude [?]")

    axes[1].plot(tt, pha[0, 0], label="phase")
    axes[1].legend()

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Phase [radian]")

    # plt.show()
    mngs.io.save(fig, CONFIG["SDIR"] + "traces.png")

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
/home/ywatanabe/proj/entrance/mngs/dsp/_hilbert.py
"""
