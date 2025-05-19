# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dsp/filt.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:05:47 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/dsp/filt.py
# 
# import mngs
# import numpy as np
# 
# from ..decorators import torch_fn
# from ..nn import (BandPassFilter, BandStopFilter, GaussianFilter,
#                   HighPassFilter, LowPassFilter)
# 
# 
# @torch_fn
# def gauss(x, sigma, t=None):
#     return GaussianFilter(sigma)(x, t=t)
# 
# @torch_fn
# def bandpass(x, fs, bands, t=None):
#     return BandPassFilter(bands, fs, x.shape[-1])(x, t=t)
# 
# @torch_fn
# def bandstop(x, fs, bands, t=None):
#     return BandStopFilter(bands, fs, x.shape[-1])(x, t=t)
# 
# @torch_fn
# def lowpass(x, fs, cutoffs_hz, t=None):
#     return LowPassFilter(cutoffs_hz, fs, x.shape[-1])(x, t=t)
# 
# @torch_fn
# def highpass(x, fs, cutoffs_hz, t=None):
#     return HighPassFilter(cutoffs_hz, fs, x.shape[-1])(x, t=t)
# 
# def _custom_print(x):
#     print(type(x), x.shape)
# 
# if __name__ == "__main__":
#     import sys
# 
#     import matplotlib.pyplot as plt
#     import torch
# 
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
# 
#     # Parametes
#     T_SEC = 1
#     SRC_FS = 1024
#     FREQS_HZ = list(np.linspace(0, 500, 10, endpoint=False).astype(int))
#     SIG_TYPE = "periodic"
#     BANDS = np.vstack([[80, 310]])
#     SIGMA = 3
# 
#     # Demo Signal
#     xx, tt, fs = mngs.dsp.demo_sig(
#         t_sec=T_SEC,
#         fs=SRC_FS,
#         freqs_hz=FREQS_HZ,
#         sig_type=SIG_TYPE,
#     )
# 
#     # Filtering
#     x_bp, t_bp = mngs.dsp.filt.bandpass(xx, fs, BANDS, t=tt)
#     x_bs, t_bs = mngs.dsp.filt.bandstop(xx, fs, BANDS, t=tt)
#     x_lp, t_lp = mngs.dsp.filt.lowpass(xx, fs, BANDS[:, 0], t=tt)
#     x_hp, t_hp = mngs.dsp.filt.highpass(xx, fs, BANDS[:, 1], t=tt)
#     x_g, t_g = mngs.dsp.filt.gauss(xx, sigma=SIGMA, t=tt)
#     filted = {
#         f"Original (Sum of {FREQS_HZ}-Hz signals)": (xx, tt, fs),
#         f"Bandpass-filtered ({BANDS[0][0]} - {BANDS[0][1]} Hz)": (
#             x_bp,
#             t_bp,
#             fs,
#         ),
#         f"Bandstop-filtered ({BANDS[0][0]} - {BANDS[0][1]} Hz)": (
#             x_bs,
#             t_bs,
#             fs,
#         ),
#         f"Lowpass-filtered ({BANDS[0][0]} Hz)": (x_lp, t_lp, fs),
#         f"Highpass-filtered ({BANDS[0][1]} Hz)": (x_hp, t_hp, fs),
#         f"Gaussian-filtered (sigma = {SIGMA} SD [point])": (x_g, t_g, fs),
#     }
# 
#     # Plots traces
#     fig, axes = plt.subplots(
#         nrows=len(filted), ncols=1, sharex=True, sharey=True
#     )
#     i_batch = 0
#     i_ch = 0
#     i_filt = 0
#     for ax, (k, v) in zip(axes, filted.items()):
#         _xx, _tt, _fs = v
#         if _xx.ndim == 3:
#             _xx = _xx[i_batch, i_ch]
#         elif _xx.ndim == 4:
#             _xx = _xx[i_batch, i_ch, i_filt]
#         ax.plot(_tt, _xx, label=k)
#         ax.legend(loc="upper left")
# 
#     fig.suptitle("Filtered")
#     fig.supxlabel("Time [s]")
#     fig.supylabel("Amplitude")
# 
#     mngs.io.save(fig, "traces.png")
# 
#     # Calculates and Plots PSD
#     fig, axes = plt.subplots(
#         nrows=len(filted), ncols=1, sharex=True, sharey=True
#     )
#     i_batch = 0
#     i_ch = 0
#     i_filt = 0
#     for ax, (k, v) in zip(axes, filted.items()):
#         _xx, _tt, _fs = v
# 
#         _psd, ff = mngs.dsp.psd(_xx, _fs)
#         if _psd.ndim == 3:
#             _psd = _psd[i_batch, i_ch]
#         elif _psd.ndim == 4:
#             _psd = _psd[i_batch, i_ch, i_filt]
# 
#         ax.plot(ff, _psd, label=k)
#         ax.legend(loc="upper left")
# 
#         for bb in np.hstack(BANDS):
#             ax.axvline(x=bb, color=CC["grey"], linestyle="--")
# 
#     fig.suptitle("PSD (power spectrum density) of filtered signals")
#     fig.supxlabel("Frequency [Hz]")
#     fig.supylabel("log(Power [uV^2 / Hz]) [a.u.]")
#     mngs.io.save(fig, "psd.png")
# 
#     # Close
#     mngs.gen.close(CONFIG)
# 
# # EOF
# 
# """
# /home/ywatanabe/proj/mngs/src/mngs/dsp/filt.py
# """
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dsp/filt.py
# --------------------------------------------------------------------------------
