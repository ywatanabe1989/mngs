#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-12 09:59:59 (ywatanabe)"

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

    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    # Parameters
    FS = 1024
    FREQS_HZ = [30, 100, 250]
    SIG_TYPE = "tensorpac"
    T_SEC = 4

    # Demo signal
    xx, tt, fs = mngs.dsp.demo_sig(
        batch_size=2,
        n_chs=2,
        n_segments=2,
        fs=FS,
        freqs_hz=FREQS_HZ,
        sig_type=SIG_TYPE,
    )

    if SIG_TYPE == "tensorpac":
        xx = xx[:, :, 0, :]  # The first segment

    # Main
    pha, amp, freqs = wavelet(xx, fs, device="cuda")

    # Plots
    fig, axes = mngs.plt.subplots(nrows=3, ncols=1)

    # Trace
    axes[0].plot(tt, xx[0, 0], label="orig")
    axes[0].set_ylabel("Amplitude")
    # Phase
    axes[1].imshow2d(pha[0, 0].T, cbar_label="Phase")
    axes[1] = mngs.plt.ax.set_ticks(axes[1], xticks=tt, yticks=freqs)
    axes[1].set_ylabel("Frequency [Hz]")
    # Amplitude
    axes[2].imshow2d(np.log(amp[0, 0] + 1e-5).T, cbar_label="Amplitude")
    axes[2] = mngs.plt.ax.set_ticks(axes[2], xticks=tt, yticks=freqs)
    axes[2].set_ylabel("Frequency [Hz]")

    for ax in axes:
        ax.legend(loc="upper left")
        ax = mngs.plt.ax.set_n_ticks(ax)

    fig.suptitle("Wavelet Transformation")
    fig.supxlabel("Time [s]")

    mngs.io.save(fig, CONFIG["SDIR"] + "wavelet.png")
    # plt.show()

    # # Plots phase
    # _freqs = freqs[freqs <= 20]
    # fig, axes = mngs.plt.subplots(nrows=len(_freqs), sharex=True)
    # for ax, (i_ff, ff) in zip(axes, enumerate(_freqs)):
    #     ax.plot(tt, pha[0, 0, i_ff], label=f"{ff:.1f} Hz")
    #     ax.legend(loc="upper left")
    # plt.show()

    # # Plots phase from tensorpac
    # (
    #     phases,
    #     amplitudes,
    #     freqs_pha,
    #     freqs_amp,
    #     pac,
    # ) = mngs.dsp.utils.pac.calc_pac_with_tensorpac(xx, fs, T_SEC)

    # _freqs = freqs_pha[freqs_pha <= 20]
    # fig, axes = mngs.plt.subplots(nrows=len(_freqs), sharex=True)
    # for ax, (i_ff, ff) in zip(axes, enumerate(_freqs)):
    #     ax.plot(tt, phases[i_ff, 0], label=f"{ff:.1f} Hz")
    #     ax.legend(loc="upper left")
    # plt.show()

    # tplot(tt, pha[0, 0][i_ff][:10], label=f"{ff:.1f} Hz")

    # ax.plot(tt, pha[0, 0][i_ff], label=f"{ff:.1f} Hz")
    # ax.legend(loc="upper left")
    # plt.show

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
/home/ywatanabe/proj/entrance/mngs/dsp/_wavelet.py
"""
