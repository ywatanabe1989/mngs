"""
This script does XYZ.
"""

# Imports
import sys

import matplotlib.pyplot as plt
import mngs
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

    # ## Plot
    # fig, ax = plt.subplots()
    # ax = p.comodulogram(
    #     pac, title=p.method.replace(" (", f" ({k})\n("), cmap="viridis"
    # )
    # ax = mngs.plt.ax.set_n_ticks(ax)
    freqs_amp = p.f_amp.mean(axis=-1)
    freqs_pha = p.f_pha.mean(axis=-1)

    pac = pac.T  # (amp, pha) -> (pha, amp)

    return phases, amplitudes, freqs_pha, freqs_amp, pac
    # return phases and amplitudes for future use in my implementation
    # as the aim of this code is to confirm the calculation of Modulation Index only
    # without considering bandpass filtering and hilbert transformation.


def plot_PAC_mngs_vs_tensorpac(pac_mngs, pac_tp, freqs_pha, freqs_amp):
    assert pac_mngs.shape == pac_tp.shape

    # Plots
    fig, axes = mngs.plt.subplots(ncols=3)

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

    for ax in axes:
        ax = mngs.plt.ax.set_ticks(
            ax, xticks=freqs_pha.astype(int), yticks=freqs_amp.astype(int)
        )
        ax = mngs.plt.ax.set_n_ticks(ax)

    fig.suptitle("PAC (MI) values")
    fig.supxlabel("Frequency for phase [Hz]")
    fig.supylabel("Frequency for amplitude [Hz]")

    return fig


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    # Parameters
    FS = 512
    T_SEC = 8

    xx, tt, fs = mngs.dsp.demo_sig(
        batch_size=4,
        n_chs=19,
        n_segments=2,
        fs=FS,
        t_sec=T_SEC,
        sig_type="tensorpac",
    )
    xx = torch.tensor(xx)

    phases, amplitudes, freqs_pha, freqs_amp, pac = calc_pac_with_tensorpac(
        xx, fs, T_SEC, i_batch=0, i_ch=0
    )

    phases.shape

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
/ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/dsp/utils/_calc_pac_with_tensorpac.py
"""
