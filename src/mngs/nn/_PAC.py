#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-08 12:28:28 (ywatanabe)"


import warnings

import torch
import torch.nn as nn
from mngs.nn import BandPassFilter, Hilbert, ModulationIndex


class PAC(nn.Module):
    def __init__(
        self,
        fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=100,
        amp_start_hz=60,
        amp_end_hz=160,
        amp_n_bands=100,
    ):
        super().__init__()

        # Bands definitions
        self.BANDS_PHA = self.calc_bands_pha(
            start_hz=pha_start_hz,
            end_hz=pha_end_hz,
            n_bands=pha_n_bands,
        )
        self.BANDS_AMP = self.calc_bands_amp(
            start_hz=amp_start_hz,
            end_hz=amp_end_hz,
            n_bands=amp_n_bands,
        )

        # Calculation modules
        self.bandpass_filters_pha = [
            BandPassFilter(ll, hh, fs, kernel_size=None)
            for ll, hh in self.BANDS_PHA
        ]
        self.bandpass_filters_amp = [
            BandPassFilter(ll, hh, fs, kernel_size=None)
            for ll, hh in self.BANDS_AMP
        ]
        self.hilbert = Hilbert(dim=-1)
        self.modulation_index = ModulationIndex(n_bins=18)

    def forward(self, x):
        """x.shape: (batch_size, n_chs, seq_len) or (batch_size, n_chs, n_segments, seq_len)"""

        x = self._ensure_4d_input(x)
        # Since seq_len will be truncated after bandpassfiltering, seq_len should be only used here.
        batch_size, n_chs, n_segments, _seq_len = x.shape
        x_3d = x.reshape(batch_size * n_chs, n_segments, _seq_len)

        I_BATCH_SIZE = 0
        I_CHS = 1
        I_FREQS = 2
        I_SEGMENTS = 3
        I_SEQ_LEN = 4

        # Phase
        pha = torch.stack(
            [bf_pha(x_3d) for bf_pha in self.bandpass_filters_pha],
            dim=I_FREQS - 1,
        )
        pha = self.hilbert(pha)[..., 0]
        pha = pha.reshape(batch_size, n_chs, *pha.shape[1:])
        assert pha.ndim == 5

        # Amplitude
        amp = torch.stack(
            [bf_amp(x_3d) for bf_amp in self.bandpass_filters_amp],
            dim=I_FREQS - 1,
        )
        amp = self.hilbert(amp)[..., 1]
        amp = amp.reshape(batch_size, n_chs, *amp.shape[1:])
        assert amp.ndim == 5

        # pha = (
        #     torch.stack(
        #         [
        #             self.hilbert(bf_pha(x.squeeze(0)))
        #             for bf_pha in self.bandpass_filters_pha
        #         ],
        #         dim=I_FREQS,
        #     )[..., 0]
        #     .unsqueeze(I_CHS)
        #     .transpose(I_FREQS, I_SEGMENTS)
        # )
        # amp = (
        #     torch.stack(
        #         [
        #             self.hilbert(bf_amp(x.squeeze(0)))
        #             for bf_amp in self.bandpass_filters_amp
        #         ],
        #         dim=I_FREQS,
        #     )[..., 1]
        #     .unsqueeze(I_CHS)
        #     .transpose(I_FREQS, I_SEGMENTS)
        # )
        pac = self.modulation_index(pha, amp)
        return pac

    @staticmethod
    def calc_bands_pha(start_hz=2, end_hz=20, n_bands=100):
        start_hz = start_hz if start_hz is not None else 2
        end_hz = end_hz if end_hz is not None else 20
        mid_hz = torch.linspace(start_hz, end_hz, n_bands)
        return torch.cat(
            (
                mid_hz.unsqueeze(1) - mid_hz.unsqueeze(1) / 4.0,
                mid_hz.unsqueeze(1) + mid_hz.unsqueeze(1) / 4.0,
            ),
            dim=1,
        )  # [REVISED]
        # return torch.cat[mid_hz - mid_hz / 4.0, mid_hz + mid_hz / 4.0]
        # mid_hz = np.linspace(start_hz, end_hz, n_bands)
        # return np.c_[mid_hz - mid_hz / 4.0, mid_hz + mid_hz / 4.0]

    @staticmethod
    def calc_bands_amp(start_hz=30, end_hz=160, n_bands=100):
        start_hz = start_hz if start_hz is not None else 30
        end_hz = end_hz if end_hz is not None else 160
        mid_hz = torch.linspace(start_hz, end_hz, n_bands)
        return torch.cat(
            (
                mid_hz.unsqueeze(1) - mid_hz.unsqueeze(1) / 8.0,
                mid_hz.unsqueeze(1) + mid_hz.unsqueeze(1) / 8.0,
            ),
            dim=1,
        )  # [REVISED]

        # mid_hz = torch.linspace(start_hz, end_hz, n_bands)
        # return torch.c_[mid_hz - mid_hz / 8.0, mid_hz + mid_hz / 8.0]

        # mid_hz = np.linspace(start_hz, end_hz, n_bands)
        # return np.c_[mid_hz - mid_hz / 8.0, mid_hz + mid_hz / 8.0]

    @staticmethod
    def _ensure_4d_input(x):
        if x.ndim != 4:
            message = f"Input tensor must be 4D with the shape (batch_size, n_chs, n_segments, seq_len). Received shape: {x.shape}"

        if x.ndim == 3:
            warnings.warn(
                "'n_segments' was determined to be 1, assuming your input is (batch_size, n_chs, seq_len).",
                UserWarning,
            )
            x = x.unsqueeze(-2)

        if x.ndim != 4:
            raise ValueError(message)

        return x


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-08 12:51:04 (ywatanabe)"

# import warnings

# import torch
# import torch.nn as nn
# from mngs.nn import BandPassFilter, Hilbert, ModulationIndex


# class PAC(nn.Module):
#     def __init__(
#         self,
#         fs,
#         pha_start_hz=2,
#         pha_end_hz=20,
#         pha_n_bands=100,
#         amp_start_hz=60,
#         amp_end_hz=160,
#         amp_n_bands=100,
#     ):
#         super().__init__()

#         # Bands definitions
#         self.BANDS_PHA = self.calc_bands_pha(
#             start_hz=pha_start_hz,
#             end_hz=pha_end_hz,
#             n_bands=pha_n_bands,
#         )
#         self.BANDS_AMP = self.calc_bands_amp(
#             start_hz=amp_start_hz,
#             end_hz=amp_end_hz,
#             n_bands=amp_n_bands,
#         )

#         # Calculation modules
#         self.bandpass_filters_pha = [
#             BandPassFilter(ll, hh, fs, kernel_size=None)
#             for ll, hh in self.BANDS_PHA
#         ]
#         self.bandpass_filters_amp = [
#             BandPassFilter(ll, hh, fs, kernel_size=None)
#             for ll, hh in self.BANDS_AMP
#         ]
#         self.hilbert = Hilbert(dim=-1)
#         self.modulation_index = ModulationIndex(n_bins=18)

#     def forward(self, x):
#         """x.shape: (batch_size, n_chs, seq_len) or (batch_size, n_chs, n_segments, seq_len)"""

#         x = self._ensure_4d_input(x)
#         # Since seq_len will be truncated after bandpassfiltering, seq_len should be only used here.
#         batch_size, n_chs, n_segments, _seq_len = x.shape
#         x_3d = x.reshape(batch_size * n_chs, n_segments, _seq_len)

#         I_BATCH_SIZE = 0
#         I_CHS = 1
#         I_FREQS = 2
#         I_SEGMENTS = 3
#         I_SEQ_LEN = 4

#         # Phase
#         pha = torch.stack(
#             [bf_pha(x_3d) for bf_pha in self.bandpass_filters_pha],
#             dim=I_FREQS - 1,
#         )
#         pha = self.hilbert(pha)[..., 0]
#         pha = pha.reshape(batch_size, n_chs, *pha.shape[1:])
#         assert pha.ndim == 5

#         # Amplitude
#         amp = torch.stack(
#             [bf_amp(x_3d) for bf_amp in self.bandpass_filters_amp],
#             dim=I_FREQS - 1,
#         )
#         amp = self.hilbert(amp)[..., 1]
#         amp = amp.reshape(batch_size, n_chs, *amp.shape[1:])
#         assert amp.ndim == 5

#         # pha = (
#         #     torch.stack(
#         #         [
#         #             self.hilbert(bf_pha(x.squeeze(0)))
#         #             for bf_pha in self.bandpass_filters_pha
#         #         ],
#         #         dim=I_FREQS,
#         #     )[..., 0]
#         #     .unsqueeze(I_CHS)
#         #     .transpose(I_FREQS, I_SEGMENTS)
#         # )
#         # amp = (
#         #     torch.stack(
#         #         [
#         #             self.hilbert(bf_amp(x.squeeze(0)))
#         #             for bf_amp in self.bandpass_filters_amp
#         #         ],
#         #         dim=I_FREQS,
#         #     )[..., 1]
#         #     .unsqueeze(I_CHS)
#         #     .transpose(I_FREQS, I_SEGMENTS)
#         # )
#         pac = self.modulation_index(pha, amp)
#         return pac

#     @staticmethod
#     def calc_bands_pha(start_hz=2, end_hz=20, n_bands=100):
#         start_hz = start_hz if start_hz is not None else 2
#         end_hz = end_hz if end_hz is not None else 20
#         # n_bands = dict(lres=10, mres=30, hres=50, demon=70, hulk=100)[res_str]
#         mid_hz = np.linspace(start_hz, end_hz, n_bands)
#         return np.c_[mid_hz - mid_hz / 4.0, mid_hz + mid_hz / 4.0]

#     @staticmethod
#     def calc_bands_amp(start_hz=30, end_hz=160, n_bands=100):
#         start_hz = start_hz if start_hz is not None else 30
#         end_hz = end_hz if end_hz is not None else 160
#         # n_bands = dict(lres=10, mres=30, hres=50, demon=70, hulk=100)[res_str]
#         mid_hz = np.linspace(start_hz, end_hz, n_bands)
#         return np.c_[mid_hz - mid_hz / 8.0, mid_hz + mid_hz / 8.0]

#     @staticmethod
#     def _ensure_4d_input(x):
#         if x.ndim != 4:
#             message = f"Input tensor must be 4D with the shape (batch_size, n_chs, n_segments, seq_len). Received shape: {x.shape}"

#         if x.ndim == 3:
#             warnings.warn(
#                 "'n_segments' was determined to be 1, assuming your input is (batch_size, n_chs, seq_len).",
#                 UserWarning,
#             )
#             x = x.unsqueeze(-2)

#         if x.ndim != 4:
#             raise ValueError(message)

#         return x
