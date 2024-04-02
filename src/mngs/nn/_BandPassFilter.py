#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-01 18:16:03 (ywatanabe)"

import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

# class BandPassFilter(nn.Module):
#     def __init__(self, samp_rate, low_hz=30, high_hz=60):
#         super().__init__()
#         self.samp_rate = samp_rate
#         self.low_hz = low_hz
#         self.high_hz = high_hz
#         self.order = 4  # Assuming a 4th order filter for simplicity
#         nyq = 0.5 * samp_rate
#         low = low_hz / nyq
#         high = high_hz / nyq
#         b, a = scipy.signal.butter(self.order, [low, high], btype="band")
#         self.b = (
#             torch.tensor(b, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#         )  # [REVISED]
#         self.a = (
#             torch.tensor(a, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#         )  # [REVISED]

#     def forward(self, x):
#         """x.shape: (batch_size, n_chs, seq_len)"""
#         x = x.unsqueeze(1)  # Add channel dimension [REVISED]
#         y = F.conv1d(x, self.b, self.a, padding=self.order)  # [REVISED]
#         return y.squeeze(1)  # Remove channel dimension [REVISED]


class BandPassFilter(nn.Module):
    def __init__(self, order=None, low_hz=30, high_hz=60, fs=250):
        super().__init__()
        self.fs = fs
        self.order = fs if order is None else order
        self.numtaps = self.order + 1
        filter_npy = scipy.signal.firwin(
            self.numtaps,
            [low_hz, high_hz],
            pass_zero="bandpass",
            fs=fs,
        )

        self.register_buffer(
            "filters",
            torch.tensor(filter_npy).unsqueeze(0).unsqueeze(0),  # [REVISED]
        )

    def forward(self, x):
        if x.ndim == 3:
            bs, n_chs, sig_len = x.shape
            filters_expanded = self.filters.expand(1, -1, -1).repeat(
                n_chs, 1, 1
            )  # [REVISED]
            x = x.reshape(bs * n_chs, 1, sig_len)
        else:
            filters_expanded = self.filters

        filted = F.conv1d(
            x, filters_expanded.type_as(x), padding=int(self.numtaps / 2)
        )
        filted = filted.flip(dims=[-1])
        filted = F.conv1d(
            filted, filters_expanded.type_as(x), padding=int(self.numtaps / 2)
        )
        filted = filted.flip(dims=[-1])

        filted = filted[..., 1:-1]

        if x.ndim == 3:
            filted = filted.reshape(bs, n_chs, -1)

        return filted


# # class BandPassFilter(nn.Module):
# #     def __init__(self, fs, low_hz, high_hz, order=None):
# #         super().__init__()
# #         self.fs = fs
# #         self.order = fs if order is None else order
# #         self.numtaps = self.order + 1
# #         filter_npy = scipy.signal.firwin(
# #             self.numtaps,
# #             [low_hz, high_hz],
# #             pass_zero="bandpass",
# #             fs=fs,
# #         )

# #         # Register the filter as a buffer
# #         self.register_buffer(
# #             "filters", torch.tensor(filter_npy).unsqueeze(0).unsqueeze(0)
# #         )

# #     def forward(self, x):
# #         # Determine the number of channels from the input tensor
# #         if x.ndim == 3:
# #             bs, n_chs, sig_len = x.shape
# #             # Expand the filter to match the number of channels in the input
# #             filters_expanded = self.filters.expand(n_chs, -1, -1)
# #             # Reshape x to combine batch size and channels for convolution
# #             x = x.reshape(bs * n_chs, 1, sig_len)
# #         else:
# #             # If not a 3D tensor, assume it's already in the correct shape for convolution
# #             filters_expanded = self.filters

# #         # Perform the convolution with the expanded filters
# #         filted = F.conv1d(
# #             x, filters_expanded.type_as(x), padding=int(self.numtaps / 2)
# #         )
# #         filted = filted.flip(dims=[-1])  # to backward
# #         filted = F.conv1d(
# #             filted, filters_expanded.type_as(x), padding=int(self.numtaps / 2)
# #         )
# #         filted = filted.flip(dims=[-1])  # reverse to the original order

# #         # Adjust the output to remove the extra samples added by padding
# #         filted = filted[..., 1:-1]

# #         # If the input was a 3D tensor, reshape the output to match the input shape
# #         if x.ndim == 3:
# #             filted = filted.reshape(bs, n_chs, -1)

# #         return filted


# # # class BandPassFilter(nn.Module):
# # #     def __init__(self, order=None, low_hz=30, high_hz=60, fs=250, n_chs=19):
# # #         super().__init__()
# # #         self.fs = fs
# # #         self.order = fs if order is None else order
# # #         self.numtaps = self.order + 1
# # #         filter_npy = scipy.signal.firwin(
# # #             self.numtaps,
# # #             [low_hz, high_hz],
# # #             pass_zero="bandpass",
# # #             fs=fs,
# # #         )

# # #         self.register_buffer(
# # #             "filters", torch.tensor(filter_npy).unsqueeze(0).unsqueeze(0)
# # #         )

# # #     def forward(self, x):
# # #         dim = x.ndim

# # #         if dim == 3:
# # #             bs, n_chs, sig_len = x.shape
# # #             x = x.reshape(bs * n_chs, 1, sig_len)

# # #         filted = F.conv1d(
# # #             x, self.filters.type_as(x), padding=int(self.numtaps / 2)
# # #         )
# # #         filted = filted.flip(dims=[-1])  # to backward
# # #         filted = F.conv1d(
# # #             filted, self.filters.type_as(x), padding=int(self.numtaps / 2)
# # #         )
# # #         filted = filted.flip(dims=[-1])  # reverse to the original order

# # #         filted = filted[..., 1:-1]

# # #         if dim == 3:
# # #             filted = filted.reshape(bs, n_chs, -1)

# # #         return filted
