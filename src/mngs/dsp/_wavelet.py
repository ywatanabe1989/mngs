#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-13 00:51:35 (ywatanabe)"

"""
This script does XYZ.
"""

import mngs
from mngs.general import torch_fn
from mngs.nn import Wavelet


# Functions
@torch_fn
def wavelet(
    x,
    fs,
    freq_scale="linear",
    out_scale="linear",
    device="cuda",
):
    m = Wavelet(fs, freq_scale=freq_scale, out_scale=out_scale)
    pha, amp, freqs = m(x)
    return pha, amp, freqs


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import numpy as np

    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, agg=True
    )

    # Parameters
    FS = 512
    SIG_TYPE = "chirp"
    T_SEC = 4

    # Demo signal
    xx, tt, fs = mngs.dsp.demo_sig(
        batch_size=2,
        n_chs=2,
        n_segments=2,
        t_sec=T_SEC,
        fs=FS,
        sig_type=SIG_TYPE,
    )

    if SIG_TYPE in ["tensorpac", "pac"]:
        i_segment = 0
        xx = xx[:, :, i_segment, :]

    # Main
    pha, amp, freqs = wavelet(xx, fs, device="cuda")

    # Plots
    i_batch, i_ch = 0, 0
    fig, axes = mngs.plt.subplots(nrows=3)

    # # Time vector for x-axis extents
    # time_extent = [tt.min(), tt.max()]

    # Trace
    axes[0].plot(tt, xx[i_batch, i_ch], label=SIG_TYPE)
    axes[0].set_ylabel("Amplitude [?V]")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Signal")

    # Amplitude
    # extent = [time_extent[0], time_extent[1], freqs.min(), freqs.max()]
    axes[1].imshow2d(
        np.log(amp[i_batch, i_ch] + 1e-5).T,
        cbar_label="Log(amplitude [?V]) [a.u.]",
        aspect="auto",
        # extent=extent,
        # origin="lower",
    )
    axes[1] = mngs.plt.ax.set_ticks(axes[1], xticks=tt, yticks=freqs)
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_title("Amplitude")

    # Phase
    axes[2].imshow2d(
        pha[i_batch, i_ch].T,
        cbar_label="Phase [rad]",
        aspect="auto",
        # extent=extent,
        # origin="lower",
    )
    axes[2] = mngs.plt.ax.set_ticks(axes[2], xticks=tt, yticks=freqs)
    axes[2].set_ylabel("Frequency [Hz]")
    axes[2].set_title("Phase")

    fig.suptitle("Wavelet Transformation")
    fig.supxlabel("Time [s]")

    for ax in axes:
        ax = mngs.plt.ax.set_n_ticks(ax)
        # ax.set_xlim(time_extent[0], time_extent[1])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    mngs.io.save(fig, "wavelet.png")

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
/home/ywatanabe/proj/entrance/mngs/dsp/_wavelet.py
"""
