#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-12 23:29:16 (ywatanabe)"

"""
This script does XYZ.
"""

# Imports
import sys
import warnings

import matplotlib.pyplot as plt
import mngs
import torch
import torch.nn as nn

# Config
CONFIG = mngs.gen.load_configs()

# Functions
class PAC(nn.Module):
    def __init__(
        self,
        x_shape,
        fs,
        pha_start_hz=2,
        pha_end_hz=20,
        pha_n_bands=50,
        amp_start_hz=60,
        amp_end_hz=160,
        amp_n_bands=30,
        fp16=False,
        trainable=False,
    ):
        super().__init__()

        # A static, general purpose BandPassFilter
        if not trainable:
            # First, bands definitions for phase and amplitude are declared
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
            bands_all = torch.vstack([self.BANDS_PHA, self.BANDS_AMP])

            # Instanciation of the static bandpass filter module
            self.bandpass = mngs.nn.BandPassFilter(
                bands_all,
                fs,
                x_shape,
                fp16=fp16,
            )
            self.PHA_MIDS_HZ = self.BANDS_PHA.mean(-1)
            self.AMP_MIDS_HZ = self.BANDS_AMP.mean(-1)

        # A trainable BandPassFilter specifically for PAC calculation. Bands will be optimized.
        else:
            self.bandpass = mngs.nn.DifferentiableBandPassFilter(
                x_shape[-1],
                fs,
                fp16=fp16,
                pha_low_hz=pha_start_hz,
                pha_high_hz=pha_end_hz,
                pha_n_bands=pha_n_bands,
                amp_low_hz=amp_start_hz,
                amp_high_hz=amp_end_hz,
                amp_n_bands=amp_n_bands,
            )
            self.PHA_MIDS_HZ = self.bandpass.pha_mids
            self.AMP_MIDS_HZ = self.bandpass.amp_mids
        self.hilbert = mngs.nn.Hilbert(dim=-1)
        self.modulation_index = mngs.nn.ModulationIndex(n_bins=18, fp16=fp16)

    @property
    def dimensions(self):
        return dict(
            I_BATCH_SIZE=0,
            I_CHS=1,
            I_FREQS=2,
            I_SEGMENTS=10,
            I_SEQ_LEN=4,
        )

    def forward(self, x):
        """x.shape: (batch_size, n_chs, seq_len) or (batch_size, n_chs, n_segments, seq_len)"""

        x = self._ensure_4d_input(x)
        # (batch_size, n_chs, n_segments, seq_len)

        batch_size, n_chs, n_segments, seq_len = x.shape

        x = x.reshape(batch_size * n_chs, n_segments, seq_len)
        # (batch_size * n_chs, n_segments, seq_len)

        x = self.bandpass(x, edge_len=0)
        # (batch_size*n_chs, n_segments, n_pha_bands + n_amp_bands, seq_len)

        x = self.hilbert(x)
        # (batch_size*n_chs, n_segments, n_pha_bands + n_amp_bands, seq_len, pha + amp)

        x = x.reshape(batch_size, n_chs, *x.shape[1:])
        # (batch_size, n_chs, n_segments, n_pha_bands + n_amp_bands, seq_len, pha + amp)

        x = x.transpose(2, 3)
        # (batch_size, n_chs, n_pha_bands + n_amp_bands, n_segments, pha + amp)

        pha = x[:, :, : len(self.PHA_MIDS_HZ), :, :, 0]
        # (batch_size, n_chs, n_freqs_pha, n_segments, sequence_length)

        amp = x[:, :, -len(self.AMP_MIDS_HZ) :, :, :, 1]
        # (batch_size, n_chs, n_freqs_amp, n_segments, sequence_length)()

        edge_len = int(pha.shape[-1] // 8)
        pha = pha[..., edge_len:-edge_len]
        amp = amp[..., edge_len:-edge_len]

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
        )

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
        )

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
        # sig_type="tensorpac",
        sig_type="pac",
    )
    xx = torch.tensor(xx)

    # Tensorpac
    i_batch, i_ch = 0, 0
    (
        _phases,
        _amplitudes,
        _freqs_pha,
        _freqs_amp,
        pac_tp,
    ) = mngs.dsp.utils.pac.calc_pac_with_tensorpac(
        xx, fs, t_sec=T_SEC, i_batch=0, i_ch=0
    )
    # wavelet

    # mngs
    pac_mngs, pha_bands, amp_bands = mngs.dsp.pac(
        xx,
        fs,
        pha_n_bands=50,
        amp_n_bands=30,
        device="cpu",
        fp16=False,
        trainable=True,
    )

    assert torch.allclose(torch.tensor(_freqs_pha).float(), pha_bands)
    assert torch.allclose(torch.tensor(_freqs_amp).float(), amp_bands)

    # pac_mngs.sum().backward()  # OK

    pac_mngs = pac_mngs.detach().cpu().numpy()
    pha_bands = pha_bands.detach().cpu().numpy()
    amp_bands = amp_bands.detach().cpu().numpy()

    def min_max(x):
        MM = x.max()
        mm = x.min()
        return (x - mm) / (MM - mm)

    fig = mngs.dsp.utils.pac.plot_PAC_mngs_vs_tensorpac(
        # min_max(pac_mngs[i_batch, i_ch]),
        # min_max(pac_tp),
        pac_mngs[i_batch, i_ch],
        pac_tp,
        # np.log(pac_mngs[i_batch, i_ch] + 1e-5),
        # np.log(pac_tp + 1e-5),
        pha_bands.astype(int),
        amp_bands.astype(int),
    )
    mngs.io.save(fig, CONFIG["SDIR"] + "pac.png")
    plt.show()

    # # Close
    # mngs.gen.close(CONFIG)

    """
    # Speed check

    # It takes 11.5 ms to calculate PAC from 20 x 4-second segments at 512 Hz -> PAC (50, 30) = (n_phase_bands, n_amp_bands)

    # xx.shape (1, 1, 20, 2048)
    # fs # 512

    # Including instanciation of the PAC module
    %timeit mngs.dsp.pac(xx, fs, pha_n_bands=50, amp_n_bands=30, device="cuda")
    # 26.4 ms +- 81.9 us per loop (mean +- std. dev. of 7 runs, 10 loops each)
    # pac_mngs.shape (50, 30)

    # # CPU
    # mngs.dsp.pac(torch.tensor(xx).float(), fs, pha_n_bands=50, amp_n_bands=30, device="cpu")
    # %timeit mngs.dsp.pac(torch.tensor(xx).float(), fs, pha_n_bands=50, amp_n_bands=30, device="cpu")



    # PAC calculation with mngs on cuda
    xx = torch.tensor(xx).cuda().float()
    m = PAC(xx.shape, fs, fp16=False, trainable=True).cuda()
    mngs.ml.utils.check_params(m)
    pac = m(xx)
    %timeit pac = m(xx)
    pac.shape # (4, 19, 50, 30)


    | float | 13 GB | 44.5 ms |
    | half  |  7 GB | 28.5 ms |


    # PAC calculation with mngs on cuda
    xx.shape # (4, 19, 1, 2048)
    fs # 512
    xx = torch.tensor(xx).float()
    m = PAC(xx.shape, fs, fp16=False, trainable=True).cpu()
    %timeit pac = m(xx)
    pac.shape # (4, 19, 50, 30)

    | half  | 0 GB | 1,970 ms |
    | float | 0 GB |   931 ms |


    VRAM: 0 GB at float
    VRAM: 0 GB at half
    """

# EOF

"""
/home/ywatanabe/proj/entrance/mngs/dsp/nn/_PAC.py
"""

# # close
# fig, axes = mngs.plt.subplots(ncols=2)
# axes[0].imshow2d(pac_mngs[i_batch, i_ch])
# axes[1].imshow2d(pac_tp)
# mngs.io.save(fig, CONFIG["SDIR"] + "pac.png")
# import numpy as np
# np.corrcoef(pac_mngs[i_batch, i_ch], pac_tp)[0, 1]
# import matplotlib

# plt.close("all")
# matplotlib.use("TkAgg")
# plt.scatter(pac_mngs[i_batch, i_ch].reshape(-1), pac_tp.reshape(-1))
# plt.show()
