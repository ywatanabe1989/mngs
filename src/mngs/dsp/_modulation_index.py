#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-08 10:40:23 (ywatanabe)"

import torch
from mngs.general import torch_fn
from mngs.nn import ModulationIndex


@torch_fn
def modulation_index(pha, amp, n_bins=18):
    """
    pha: (batch_size, n_chs, n_freqs_pha, n_segments, seq_len)
    amp: (batch_size, n_chs, n_freqs_amp, n_segments, seq_len)
    """
    return ModulationIndex(n_bins=n_bins)(pha, amp)


def plot_comodulogram_tensorpac(xx, fs, t_sec, ts=None):
    import tensorpac
    from tensorpac import Pac

    # Morlet's Wavelet Transfrmation
    p = tensorpac.Pac(f_pha="hres", f_amp="hres", dcomplex="wavelet")

    # Bandpass Filtering and Hilbert Transformation
    i_batch, i_ch = 0, 0
    ts("Bandpassfiltering and Hilbert Transformation with Tensorpac starts.")
    phases = p.filter(
        fs, xx[i_batch, i_ch], ftype="phase", n_jobs=1
    )  # (50, 20, 2048)
    amplitudes = p.filter(
        fs, xx[i_batch, i_ch], ftype="amplitude", n_jobs=1
    )  # (50, 20, 2048)
    ts("Bandpassfiltering and Hilbert Transformation with Tensorpac ends.")

    # Calculates xpac
    k = 2
    p.idpac = (k, 0, 0)
    ts("PAC Calculation with Tensorpac starts.")

    xpac = p.fit(phases, amplitudes)  # (50, 50, 20)
    pac = xpac.mean(axis=-1)  # (50, 50)

    ts("PAC Calculation with Tensorpac ends.")

    ## Plot
    fig, ax = plt.subplots()
    ax = p.comodulogram(
        pac, title=p.method.replace(" (", f" ({k})\n("), cmap="viridis"
    )
    ax = mngs.plt.ax.set_n_ticks(ax)
    import ipdb

    ipdb.set_trace()
    freqs_amp = p.f_amp.mean(axis=-1)
    freqs_pha = p.f_pha.mean(axis=-1)

    return phases, amplitudes, freqs_pha, freqs_amp
    # return phases and amplitudes for future use in my implementation
    # as the aim of this code is to confirm the calculation of Modulation Index only
    # without considering bandpass filtering and hilbert transformation.


@torch_fn
def reshape_pha_amp(pha, amp, batch_size=2, n_chs=4):
    pha = torch.tensor(pha).half()
    amp = torch.tensor(amp).half()
    pha = pha.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_chs, 1, 1, 1)
    amp = amp.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_chs, 1, 1, 1)
    return pha, amp


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mngs

    # Parameters
    fs = 128
    t_sec = 5

    # Timestamper
    ts = mngs.gen.TimeStamper()

    # Demo signal
    xx, tt, fs = mngs.dsp.demo_sig(fs=fs, t_sec=t_sec, sig_type="tensorpac")
    # xx.shape: (8, 19, 20, 512)

    # Tensorpac
    pha, amp, freqs_pha, freqs_amp = plot_comodulogram_tensorpac(
        xx,
        fs,
        t_sec=t_sec,
        ts=ts,
    )
    # mngs.io.save((pha, amp, freqs_pha, freqs_amp), "/tmp/out.pkl")
    # pha, amp, freqs_pha, freqs_amp = mngs.io.load("/tmp/out.pkl")

    # GPU calculation
    pha, amp = reshape_pha_amp(pha, amp)

    ts("PAC Calculation with mngs.dsp starts.")

    pac = mngs.dsp.modulation_index(pha, amp)

    ts("PAC Calculation with mngs.dsp ends.")

    ## Convert y-axis
    i_batch, i_ch = 0, 0

    fig, ax = mngs.plt.subplots()
    ax.imshow2d(
        pac[i_batch, i_ch],
        cbar_label="PAC values",
    )
    ax = mngs.plt.ax.set_ticks(
        ax, xticks=freqs_pha.astype(int), yticks=freqs_amp.astype(int)
    )
    ax = mngs.plt.ax.set_n_ticks(ax)
    ax.set_xlabel("Frequency for phase [Hz]")
    ax.set_ylabel("Frequency for amplitude [Hz]")
    ax.set_title("GPU calculation")

    plt.show()
