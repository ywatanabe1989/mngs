#! ./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-16 17:07:27"


"""
This script does XYZ.
"""

# Imports
import sys

import matplotlib.pyplot as plt

import numpy as np
import tensorpac


# Functions
def calc_pac_with_tensorpac(xx, fs, t_sec, i_batch=0, i_ch=0):
    # Morlet's Wavelet Transfrmation
    p = tensorpac.Pac(f_pha="hres", f_amp="mres", dcomplex="wavelet")

    # Bandpass Filtering and Hilbert Transformation
    phases = p.filter(
        fs, xx[i_batch, i_ch], ftype="phase", n_jobs=1
    )  # (50, 20, 2048)
    amplitudes = p.filter(
        fs, xx[i_batch, i_ch], ftype="amplitude", n_jobs=1
    )  # (50, 20, 2048)

    # Calculates xpac
    k = 2
    p.idpac = (k, 0, 0)
    xpac = p.fit(phases, amplitudes)  # (50, 50, 20)
    pac = xpac.mean(axis=-1)  # (50, 50)

    freqs_amp = p.f_amp.mean(axis=-1)
    freqs_pha = p.f_pha.mean(axis=-1)

    pac = pac.T  # (amp, pha) -> (pha, amp)

    return phases, amplitudes, freqs_pha, freqs_amp, pac


def plot_PAC_mngs_vs_tensorpac(pac_mngs, pac_tp, freqs_pha, freqs_amp):
    assert pac_mngs.shape == pac_tp.shape

    # Plots
    fig, axes = mngs.plt.subplots(ncols=3)  # , sharex=True, sharey=True

    # To align scalebars
    vmin = min(np.min(pac_mngs), np.min(pac_tp), np.min(pac_mngs - pac_tp))
    vmax = max(np.max(pac_mngs), np.max(pac_tp), np.max(pac_mngs - pac_tp))

    # mngs version
    ax = axes[0]
    ax.imshow2d(
        pac_mngs,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title("mngs")

    # Tensorpac
    ax = axes[1]
    ax.imshow2d(
        pac_tp,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title("Tensorpac")

    # Diff.
    ax = axes[2]
    ax.imshow2d(
        pac_mngs - pac_tp,
        cbar_label="PAC values",
        cbar_shrink=0.5,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"Difference\n(mngs - Tensorpac)")

    # for ax in axes:
    #     ax.set_ticks(
    #         x_vals=freqs_pha,
    #         # y_vals=freqs_amp,
    #     )
    #     # ax.set_n_ticks()

    fig.suptitle("PAC (MI) values")
    fig.supxlabel("Frequency for phase [Hz]")
    fig.supylabel("Frequency for amplitude [Hz]")

    return fig


if __name__ == "__main__":
    import torch

    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    # Parameters
    FS = 512
    T_SEC = 4

    xx, tt, fs = mngs.dsp.demo_sig(
        batch_size=2,
        n_chs=2,
        n_segments=2,
        fs=FS,
        t_sec=T_SEC,
        sig_type="tensorpac",
    )

    # mngs
    pac_mngs, freqs_pha, freqs_amp = mngs.dsp.pac(
        xx, fs, batch_size=2, pha_n_bands=50, amp_n_bands=30
    )
    i_batch, i_epoch = 0, 0
    pac_mngs = pac_mngs[i_batch, i_epoch]

    # Tensorpac
    phases, amplitudes, freqs_pha, freqs_amp, pac_tp = calc_pac_with_tensorpac(
        xx, fs, T_SEC, i_batch=0, i_ch=0
    )

    # Plots
    fig = plot_PAC_mngs_vs_tensorpac(pac_mngs, pac_tp, freqs_pha, freqs_amp)
    plt.show()

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
/home/ywatanabe/proj/entrance/mngs/dsp/utils/pac.py
"""
