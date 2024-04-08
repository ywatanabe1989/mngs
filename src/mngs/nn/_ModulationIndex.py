#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-08 11:30:02 (ywatanabe)"


import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mngs.general import torch_fn


class ModulationIndex(nn.Module):
    def __init__(self, n_bins=18):
        super(ModulationIndex, self).__init__()
        self.n_bins = n_bins
        self.register_buffer(
            "pha_bin_cutoffs", torch.linspace(-np.pi, np.pi, n_bins + 1)
        )

    def forward(self, pha, amp, epsilon=1e-9):
        """
        Compute the Modulation Index based on phase (pha) and amplitude (amp) tensors.

        Parameters:
        - pha (torch.Tensor): Tensor of phase values with shape
                              (batch_size, n_channels, n_freqs_pha, n_segments, sequence_length).
        - amp (torch.Tensor): Tensor of amplitude values with a similar shape as pha.
                              (batch_size, n_channels, n_freqs_amp, n_segments, sequence_length).

        Returns:
        - MI (torch.Tensor): The Modulation Index for each batch and channel.
        """
        assert pha.ndim == amp.ndim == 5

        pha, amp = pha.float().contiguous(), amp.float().contiguous()
        device = pha.device

        pha_masks = self._phase_to_masks(pha, self.pha_bin_cutoffs.to(device))
        # (batch_size, n_channels, n_freqs_pha, n_segments, sequence_length, n_bins)

        # Expands amp and masks to utilize broadcasting
        i_batch = 0
        i_chs = 1
        i_freqs_pha = 2
        i_freqs_amp = 3
        i_segments = 4
        i_time = 5
        i_bins = 6

        # Coupling
        pha_masks = pha_masks.unsqueeze(i_freqs_amp)
        amp = amp.unsqueeze(i_freqs_pha).unsqueeze(i_bins)
        amp_bins = pha_masks * amp

        # Takes mean amplitude in each bin
        amp_sums = amp_bins.sum(dim=i_time, keepdims=True)
        counts = pha_masks.sum(dim=i_time, keepdims=True)
        amp_means = amp_sums / (counts + epsilon)

        amp_probs = amp_means / (
            amp_means.sum(dim=-1, keepdims=True) + epsilon
        )

        MI = (
            torch.log(torch.tensor(self.n_bins, device=device) + epsilon)
            + (amp_probs * (amp_probs + epsilon).log()).sum(dim=-1)
        ) / (
            torch.log(torch.tensor(self.n_bins, device=device) + epsilon)
            + epsilon
        )

        MI = MI.squeeze(-1).mean(axis=-1)

        if MI.isnan().any():
            raise ValueError(
                "NaN values detected in Modulation Index calculation."
            )

        return MI

    @staticmethod
    def _phase_to_masks(pha, phase_bin_cutoffs):
        n_bins = int(len(phase_bin_cutoffs) - 1)
        bin_indices = (
            (
                (
                    torch.bucketize(pha, phase_bin_cutoffs, right=False) - 1
                ).clamp(0, n_bins - 1)
            )
            .long()
            .to(pha.device)
        )
        one_hot_masks = F.one_hot(
            bin_indices,
            num_classes=n_bins,
        )
        return one_hot_masks


def plot_comodulogram_tensorpac(xx, fs, t_sec):
    # Morlet's Wavelet Transfrmation
    p = tensorpac.Pac(f_pha="hres", f_amp="hres", dcomplex="wavelet")

    # Bandpass Filtering and Hilbert Transformation
    i_batch, i_ch = 0, 0
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

    ## Plot
    fig, ax = plt.subplots()
    ax = p.comodulogram(
        pac, title=p.method.replace(" (", f" ({k})\n("), cmap="viridis"
    )
    ax = mngs.plt.ax.set_n_ticks(ax)
    freqs_amp = p.f_amp.mean(axis=-1)
    freqs_pha = p.f_pha.mean(axis=-1)

    return phases, amplitudes, freqs_pha, freqs_amp
    # return phases and amplitudes for future use in my implementation
    # as the aim of this code is to confirm the calculation of Modulation Index only
    # without considering bandpass filtering and hilbert transformation.


def reshape_pha_amp(pha, amp, batch_size=2, n_chs=4):
    pha = torch.tensor(pha).half()
    amp = torch.tensor(amp).half()
    pha = pha.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_chs, 1, 1, 1)
    amp = amp.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_chs, 1, 1, 1)
    return pha, amp


@torch_fn
def modulation_index(pha, amp, n_bins=18):
    return ModulationIndex(n_bins=18)(pha, amp)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mngs
    import seaborn as sns
    import tensorpac
    from tensorpac import Pac
    from tqdm import tqdm

    # Parameters
    fs = 128
    t_sec = 5

    # Demo signal
    xx, tt, fs = mngs.dsp.demo_sig(fs=fs, t_sec=t_sec, sig_type="tensorpac")
    # xx.shape: (8, 19, 20, 512)

    # Tensorpac
    pha, amp, freqs_pha, freqs_amp = plot_comodulogram_tensorpac(
        xx, fs, t_sec=t_sec
    )
    # mngs.io.save((pha, amp, freqs_pha, freqs_amp), "/tmp/out.pkl")
    # pha, amp, freqs_pha, freqs_amp = mngs.io.load("/tmp/out.pkl")

    # GPU calculation
    pha, amp = reshape_pha_amp(pha, amp)
    pac = mngs.dsp.modulation_index(pha, amp)

    ## Convert y-axis
    i_batch, i_ch = 0, 0

    fig, ax = mngs.plt.subplots()
    ax.imshow2d(
        pac[i_batch, i_ch].cpu().numpy(),
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
