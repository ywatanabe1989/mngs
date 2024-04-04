#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-04 19:07:30 (ywatanabe)"

import torch
import torch.nn as nn
from mngs.general import torch_fn
from mngs.nn import PSD


@torch_fn
def psd(
    x,
    fs,
    prob=False,
    dim=-1,
):
    """
    import matplotlib.pyplot as plt
    import mngs

    x, t, fs = mngs.dsp.demo_sig()  # (batch_size, n_chs, seq_len)
    pp, ff = psd(x, fs)

    # Plots
    plt, CC = mngs.plt.configure_mpl(plt)
    fig, ax = mngs.plt.subplots()
    ax.plot(fs, pp[0, 0])
    ax.xlabel("Frequency [Hz]")
    ax.ylabel("log(Power [uV^2 / Hz]) [a.u.]")
    plt.show()
    """
    psd, freqs = PSD(fs, prob=prob, dim=dim)(x)
    return psd, freqs


def band_powers(self, psd):
    """
    Calculate the average power for specified frequency bands.
    """
    assert len(self.low_freqs) == len(self.high_freqs)

    out = []
    for ll, hh in zip(self.low_freqs, self.high_freqs):
        band_indices = torch.where((freqs >= ll) & (freqs <= hh))[0].to(
            psd.device
        )
        band_power = psd[..., band_indices].sum(dim=self.dim)
        bandwidth = hh - ll
        avg_band_power = band_power / bandwidth
        out.append(avg_band_power)
    out = torch.stack(out, dim=-1)
    return out

    # Average Power in Each Frequency Band
    avg_band_powers = self.calc_band_avg_power(psd, freqs)
    return (avg_band_powers,)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mngs

    x, t, f = mngs.dsp.demo_sig()  # (8, 19, 384)
    BANDS = mngs.dsp.PARAMS.BANDS

    pp, ff = psd(x, f, prob=True)

    plt, CC = mngs.plt.configure_mpl(plt)
    fig, ax = mngs.plt.subplots()
    ax.plot(ff, pp[0, 0])
    plt.show()
