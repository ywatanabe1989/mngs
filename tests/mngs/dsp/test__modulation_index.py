# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dsp/_modulation_index.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:09:55 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/dsp/_modulation_index.py
# 
# import torch
# 
# from ..decorators import torch_fn
# from ..nn._ModulationIndex import ModulationIndex
# 
# 
# @torch_fn
# def modulation_index(pha, amp, n_bins=18, amp_prob=False):
#     """
#     pha: (batch_size, n_chs, n_freqs_pha, n_segments, seq_len)
#     amp: (batch_size, n_chs, n_freqs_amp, n_segments, seq_len)
#     """
#     return ModulationIndex(n_bins=n_bins, amp_prob=amp_prob)(pha, amp)
# 
# def _reshape(x, batch_size=2, n_chs=4):
#     return (
#         torch.tensor(x)
#         .float()
#         .unsqueeze(0)
#         .unsqueeze(0)
#         .repeat(batch_size, n_chs, 1, 1, 1)
#     )
# 
# if __name__ == "__main__":
#     import sys
# 
#     import matplotlib.pyplot as plt
#     import mngs
# 
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, fig_scale=3
#     )
# 
#     # Parameters
#     FS = 512
#     T_SEC = 5
# 
#     # Demo signal
#     xx, tt, fs = mngs.dsp.demo_sig(fs=FS, t_sec=T_SEC, sig_type="tensorpac")
#     # xx.shape: (8, 19, 20, 512)
# 
#     # Tensorpac
#     (
#         pha,
#         amp,
#         freqs_pha,
#         freqs_amp,
#         pac_tp,
#     ) = mngs.dsp.utils.pac.calc_pac_with_tensorpac(xx, fs, t_sec=T_SEC)
# 
#     # GPU calculation with mngs.dsp.nn.ModulationIndex
#     pha, amp = _reshape(pha), _reshape(amp)
#     pac_mngs = mngs.dsp.modulation_index(pha, amp).cpu().numpy()
#     i_batch, i_ch = 0, 0
#     pac_mngs = pac_mngs[i_batch, i_ch]
# 
#     # Plots
#     fig = mngs.dsp.utils.pac.plot_PAC_mngs_vs_tensorpac(
#         pac_mngs, pac_tp, freqs_pha, freqs_amp
#     )
#     fig.suptitle("MI (modulation index) calculation")
#     mngs.io.save(fig, "modulation_index.png")
# 
#     # Close
#     mngs.gen.close(CONFIG)
# 
# # EOF
# 
# """
# /home/ywatanabe/proj/entrance/mngs/dsp/_modulation_index.py
# """
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dsp/_modulation_index.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
