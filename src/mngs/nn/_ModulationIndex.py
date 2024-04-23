#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-19 17:35:55"

"""
This script defines the ModulationIndex module.
"""

# Imports
import sys

import mngs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Functions
# dev, nan
# class ModulationIndex(nn.Module):
#     def __init__(self, n_bins=18, fp16=False, in_place=False):
#         super(ModulationIndex, self).__init__()
#         self.n_bins = n_bins
#         self.fp16 = fp16
#         self.in_place = in_place
#         self.register_buffer(
#             "pha_bin_cutoffs", torch.linspace(-np.pi, np.pi, n_bins + 1)
#         )

#     def forward(self, pha, amp, epsilon=1e-9):
#         """
#         Compute the Modulation Index based on phase and amplitude tensors.
#         """
#         assert pha.ndim == amp.ndim == 5
#         batch_size, n_chs, n_freqs_pha, n_segments, seq_len = pha.shape
#         _, _, n_freqs_amp, _, _ = amp.shape  # fixme

#         if self.fp16:
#             pha = pha.half()
#             amp = pha.half()

#         # device = pha.device
#         pha_masks = self._phase_to_masks(
#             pha, self.pha_bin_cutoffs.type_as(pha)
#         )

#         # Expands amp to use broadcasting for binning
#         amp = amp.unsqueeze(2).unsqueeze(
#             -1
#         )  # Expanded for phase frequencies and bins
#         pha_masks = pha_masks.unsqueeze(
#             3
#         )  # Expanded for amplitude frequencies

#         # Calculate the amplitude in each phase bin
#         if self.in_place:
#             amp_expanded = amp.expand(
#                 -1, -1, n_freqs_pha, -1, -1, -1, self.n_bins
#             )
#             pha_masks_expanded = pha_masks.expand(
#                 -1, -1, -1, n_freqs_amp, -1, -1, -1
#             )
#             amp_bins = pha_masks_expanded.clone().mul_(
#                 amp_expanded
#             )  # this is not wokring
#         else:
#             amp_bins = (
#                 pha_masks * amp
#             )  # this is working thanks to broadcasting

#         amp_sums = amp_bins.sum(dim=5, keepdim=True)
#         counts = pha_masks.sum(dim=5, keepdim=True)
#         amp_means = (
#             amp_sums.div_(counts.add_(epsilon))
#             if self.in_place
#             else amp_sums / (counts + epsilon)
#         )

#         amp_probs = (
#             amp_means.div_(amp_means.sum(dim=6, keepdim=True).add_(epsilon))
#             if self.in_place
#             else amp_means / (amp_means.sum(dim=6, keepdim=True) + epsilon)
#         )

#         # log_n_bins = torch.log(
#         #     torch.tensor(self.n_bins).type_as(pha) + epsilon
#         # )
#         # MI = (
#         #     log_n_bins
#         #     + (amp_probs * torch.log(amp_probs + epsilon)).sum(dim=6)
#         # ) / (log_n_bins + epsilon)
#         import ipdb

#         ipdb.set_trace()

#         MI = (
#             torch.log(torch.tensor(self.n_bins).type_as(pha) + epsilon)
#             + (amp_probs * torch.log(amp_probs + epsilon)).sum(dim=6)
#         ) / torch.log(torch.tensor(self.n_bins).type_as(pha))

#         MI = MI.mean(dim=4)  # Mean across segments

#         if MI.isnan().any():
#             raise ValueError(
#                 "NaN values detected in Modulation Index calculation."
#             )

#         return MI

#     # @staticmethod
#     # def _phase_to_masks(pha, phase_bin_cutoffs):
#     #     n_bins = int(len(phase_bin_cutoffs) - 1)
#     #     bin_indices = (
#     #         (
#     #             (
#     #                 torch.bucketize(pha, phase_bin_cutoffs, right=False) - 1
#     #             ).clamp(0, n_bins - 1)
#     #         )
#     #         .long()
#     #         .to(pha.device)
#     #     )
#     #     one_hot_masks = F.one_hot(
#     #         bin_indices,
#     #         num_classes=n_bins,
#     #     )
#     #     return one_hot_masks

#     @staticmethod
#     def _phase_to_masks(pha, phase_bin_cutoffs):
#         n_bins = len(phase_bin_cutoffs) - 1
#         pha = pha.contiguous()
#         bin_indices = torch.bucketize(pha, phase_bin_cutoffs, right=False) - 1
#         bin_indices = bin_indices.clamp(min=0, max=n_bins - 1)
#         return F.one_hot(bin_indices, num_classes=n_bins).to(
#             pha.device, dtype=pha.dtype
#         )


class ModulationIndex(nn.Module):
    def __init__(self, n_bins=18, fp16=False):
        super(ModulationIndex, self).__init__()
        self.n_bins = n_bins
        self.fp16 = fp16
        self.register_buffer(
            "pha_bin_cutoffs", torch.linspace(-np.pi, np.pi, n_bins + 1)
        )

        # self.dh_pha = mngs.gen.DimHandler()
        # self.dh_amp = mngs.gen.DimHandler()

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

        if self.fp16:
            pha, amp = pha.half().contiguous(), amp.half().contiguous()
        else:
            pha, amp = pha.float().contiguous(), amp.float().contiguous()

        device = pha.device

        pha_masks = self._phase_to_masks(pha, self.pha_bin_cutoffs.to(device))
        # (batch_size, n_channels, n_freqs_pha, n_segments, sequence_length, n_bins)

        # Expands amp and masks to utilize broadcasting
        # i_batch = 0
        # i_chs = 1
        i_freqs_pha = 2
        i_freqs_amp = 3
        # i_segments = 4
        i_time = 5
        i_bins = 6

        # Coupling
        pha_masks = pha_masks.unsqueeze(i_freqs_amp)
        amp = amp.unsqueeze(i_freqs_pha).unsqueeze(i_bins)
        amp_bins = pha_masks * amp  # this is the most memory-consuming process

        # # Batch processing to reduce maximum VRAM occupancy
        # pha_masks = self.dh_pha.fit(pha_masks, keepdims=[2, 3, 5, 6])
        # amp = self.dh_amp.fit(amp, keepdims=[2, 3, 5, 6])

        # n_chunks = len(pha_masks) // self.chunk_size
        # amp_bins = []
        # for i_chunk in range(n_chunks):
        #     start = i_chunk * self.chunk_size
        #     end = (i_chunk + 1) * self.chunk_size
        #     _amp_bins = pha_masks[start:end] * amp[start:end]
        #     amp_bins.append(_amp_bins.cpu())
        # amp_bins = torch.cat(amp_bins)
        # amp_bins = self.dh_pha.unfit(amp_bins)
        # pha_masks = self.dh_pha.unfit(pha_masks)

        # Takes mean amplitude in each bin
        amp_sums = amp_bins.sum(dim=i_time, keepdims=True).to(device)
        counts = pha_masks.sum(dim=i_time, keepdims=True)
        amp_means = amp_sums / (counts + epsilon)

        amp_probs = amp_means / (
            amp_means.sum(dim=-1, keepdims=True) + epsilon
        )

        MI = (
            torch.log(torch.tensor(self.n_bins, device=device) + epsilon)
            + (amp_probs * (amp_probs + epsilon).log()).sum(dim=-1)
        ) / torch.log(torch.tensor(self.n_bins, device=device))

        # Squeeze the n_bin dimension
        MI = MI.squeeze(-1)

        # Takes mean along the n_segments dimension
        i_segment = -1
        MI = MI.mean(axis=i_segment)

        if MI.isnan().any():
            raise ValueError(
                "NaN values detected in Modulation Index calculation."
            )

        return MI

    @staticmethod
    def _phase_to_masks(pha, phase_bin_cutoffs):
        n_bins = int(len(phase_bin_cutoffs) - 1)
        bin_indices = (
            (torch.bucketize(pha, phase_bin_cutoffs, right=False) - 1).clamp(
                0, n_bins - 1
            )
        ).long()
        one_hot_masks = (
            F.one_hot(
                bin_indices,
                num_classes=n_bins,
            )
            .bool()
            .to(pha.device)
        )
        return one_hot_masks


def _reshape(x, batch_size=2, n_chs=4):
    return (
        torch.tensor(x)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, n_chs, 1, 1, 1)
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mngs

    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, fig_scale=3
    )

    # Parameters
    FS = 512
    T_SEC = 5
    device = "cuda"

    # Demo signal
    xx, tt, fs = mngs.dsp.demo_sig(fs=FS, t_sec=T_SEC, sig_type="tensorpac")
    # xx.shape: (8, 19, 20, 512)

    # Tensorpac
    (
        pha,
        amp,
        freqs_pha,
        freqs_amp,
        pac_tp,
    ) = mngs.dsp.utils.pac.calc_pac_with_tensorpac(xx, fs, t_sec=T_SEC)

    # GPU calculation with mngs.dsp.nn.ModulationIndex
    pha, amp = _reshape(pha), _reshape(amp)

    m = ModulationIndex(n_bins=18, fp16=True).to(device)
    pac_mngs = m(pha.to(device), amp.to(device))

    # pac_mngs = mngs.dsp.modulation_index(pha, amp).cpu().numpy()
    i_batch, i_ch = 0, 0
    pac_mngs = pac_mngs[i_batch, i_ch].squeeze().numpy()

    # Plots
    fig = mngs.dsp.utils.pac.plot_PAC_mngs_vs_tensorpac(
        pac_mngs, pac_tp, freqs_pha, freqs_amp
    )
    # fig = plot_PAC_mngs_vs_tensorpac(pac_mngs, pac_tp, freqs_pha, freqs_amp)
    mngs.io.save(fig, CONFIG["SDIR"] + "modulation_index.png")  # plt.show()

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
/home/ywatanabe/proj/entrance/mngs/nn/_ModulationIndex.py
"""
