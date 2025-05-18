# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dsp/_wavelet.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:12:00 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/dsp/_wavelet.py
# 
# """mngs.dsp.wavelet function"""
# 
# 
# from ..decorators import batch_fn, torch_fn
# from ..nn._Wavelet import Wavelet
# 
# 
# # Functions
# @torch_fn
# @batch_fn
# def wavelet(
#     x,
#     fs,
#     freq_scale="linear",
#     out_scale="linear",
#     device="cuda",
#     batch_size=32,
# ):
#     m = (
#         Wavelet(fs, freq_scale=freq_scale, out_scale="linear")
#         .to(device)
#         .eval()
#     )
#     pha, amp, freqs = m(x.to(device))
# 
#     if out_scale == "log":
#         amp = (amp + 1e-5).log()
#         if amp.isnan().any():
#             print("NaN is detected while taking the lograrithm of amplitude.")
# 
#     return pha, amp, freqs
# 
# 
# # @torch_fn
# # def wavelet(
# #     x,
# #     fs,
# #     freq_scale="linear",
# #     out_scale="linear",
# #     device="cuda",
# #     batch_size=32,
# # ):
# #     @torch_fn
# #     def _wavelet(
# #         x,
# #         fs,
# #         freq_scale="linear",
# #         out_scale="linear",
# #         device="cuda",
# #     ):
# #         m = (
# #             Wavelet(fs, freq_scale=freq_scale, out_scale=out_scale)
# #             .to(device)
# #             .eval()
# #         )
# #         pha, amp, freqs = m(x.to(device))
# 
# #         if out_scale == "log":
# #             amp = (amp + 1e-5).log()
# #             if amp.isnan().any():
# #                 print(
# #                     "NaN is detected while taking the lograrithm of amplitude."
# #                 )
# 
# #         return pha, amp, freqs
# 
# #     if len(x) <= batch_size:
# #         try:
# #             pha, amp, freqs = _wavelet(
# #                 x,
# #                 fs,
# #                 freq_scale=freq_scale,
# #                 out_scale=out_scale,
# #                 device=device,
# #             )
# #             torch.cuda.empty_cache()
# #             return pha, amp, freqs
# 
# #         except Exception as e:
# #             print(e)
# #             print("\nTrying Batch Mode...")
# 
# #     n_batches = (len(x) + batch_size - 1) // batch_size
# #     device_orig = x.device
# #     pha, amp, freqs = [], [], []
# #     for i_batch in tqdm(range(n_batches)):
# #         start = i_batch * batch_size
# #         end = (i_batch + 1) * batch_size
# #         _pha, _amp, _freqs = _wavelet(
# #             x[start:end],
# #             fs,
# #             freq_scale=freq_scale,
# #             out_scale=out_scale,
# #             device=device,
# #         )
# #         torch.cuda.empty_cache()
# #         # to CPU
# #         pha.append(_pha.cpu())
# #         amp.append(_amp.cpu())
# #         freqs.append(_freqs.cpu())
# 
# #     pha = torch.vstack(pha)
# #     amp = torch.vstack(amp)
# #     freqs = freqs[0]
# 
# #     try:
# #         pha = pha.to(device_orig)
# #         amp = amp.to(device_orig)
# #         freqs = freqs.to(device_orig)
# #     except Exception as e:
# #         print(
# #             f"\nError occurred while transferring wavelet outputs back to the original device. Proceeding with CPU tensor. \n\n({e})"
# #         )
# 
# #     sleep(0.5)
# #     torch.cuda.empty_cache()
# #     return pha, amp, freqs
# 
# 
# if __name__ == "__main__":
#     import sys
# 
#     import matplotlib.pyplot as plt
#     import numpy as np
# 
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, agg=True
#     )
# 
#     # Parameters
#     FS = 512
#     SIG_TYPE = "chirp"
#     T_SEC = 4
# 
#     # Demo signal
#     xx, tt, fs = mngs.dsp.demo_sig(
#         batch_size=64,
#         n_chs=19,
#         n_segments=2,
#         t_sec=T_SEC,
#         fs=FS,
#         sig_type=SIG_TYPE,
#     )
# 
#     if SIG_TYPE in ["tensorpac", "pac"]:
#         i_segment = 0
#         xx = xx[:, :, i_segment, :]
# 
#     # Main
#     pha, amp, freqs = wavelet(xx, fs, device="cuda")
#     freqs = freqs[0, 0]
# 
#     # Plots
#     i_batch, i_ch = 0, 0
#     fig, axes = mngs.plt.subplots(nrows=3)
# 
#     # # Time vector for x-axis extents
#     # time_extent = [tt.min(), tt.max()]
# 
#     # Trace
#     axes[0].plot(tt, xx[i_batch, i_ch], label=SIG_TYPE)
#     axes[0].set_ylabel("Amplitude [?V]")
#     axes[0].legend(loc="upper left")
#     axes[0].set_title("Signal")
# 
#     # Amplitude
#     # extent = [time_extent[0], time_extent[1], freqs.min(), freqs.max()]
#     axes[1].imshow2d(
#         np.log(amp[i_batch, i_ch] + 1e-5).T,
#         cbar_label="Log(amplitude [?V]) [a.u.]",
#         aspect="auto",
#         # extent=extent,
#         # origin="lower",
#     )
#     axes[1] = mngs.plt.ax.set_ticks(axes[1], x_ticks=tt, y_ticks=freqs)
#     axes[1].set_ylabel("Frequency [Hz]")
#     axes[1].set_title("Amplitude")
# 
#     # Phase
#     axes[2].imshow2d(
#         pha[i_batch, i_ch].T,
#         cbar_label="Phase [rad]",
#         aspect="auto",
#         # extent=extent,
#         # origin="lower",
#     )
#     axes[2] = mngs.plt.ax.set_ticks(axes[2], x_ticks=tt, y_ticks=freqs)
#     axes[2].set_ylabel("Frequency [Hz]")
#     axes[2].set_title("Phase")
# 
#     fig.suptitle("Wavelet Transformation")
#     fig.supxlabel("Time [s]")
# 
#     for ax in axes:
#         ax = mngs.plt.ax.set_n_ticks(ax)
#         # ax.set_xlim(time_extent[0], time_extent[1])
# 
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# 
#     mngs.io.save(fig, "wavelet.png")
# 
#     # Close
#     mngs.gen.close(CONFIG)
# 
# # EOF
# 
# """
# /home/ywatanabe/proj/entrance/mngs/dsp/_wavelet.py
# """
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dsp/_wavelet.py
# --------------------------------------------------------------------------------
