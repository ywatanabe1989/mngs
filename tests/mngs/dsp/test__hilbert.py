# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/dsp/_hilbert.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 02:07:11 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/dsp/_hilbert.py
# 
# """
# This script does XYZ.
# """
# 
# import sys
# 
# import matplotlib.pyplot as plt
# from ..nn._Hilbert import Hilbert
# 
# from ..decorators import torch_fn
# 
# 
# # Functions
# @torch_fn
# def hilbert(
#     x,
#     dim=-1,
# ):
#     y = Hilbert(x.shape[-1], dim=dim)(x)
#     return y[..., 0], y[..., 1]
# 
# if __name__ == "__main__":
#     import mngs
# 
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
# 
#     # Parameters
#     T_SEC = 1.0
#     FS = 400
#     SIG_TYPE = "chirp"
# 
#     # Demo signal
#     xx, tt, fs = mngs.dsp.demo_sig(t_sec=T_SEC, fs=FS, sig_type=SIG_TYPE)
# 
#     # Main
#     pha, amp = hilbert(
#         xx,
#         dim=-1,
#     )
#     # (32, 19, 1280, 2)
# 
#     # Plots
#     fig, axes = mngs.plt.subplots(nrows=2, sharex=True)
#     fig.suptitle("Hilbert Transformation")
# 
#     axes[0].plot(tt, xx[0, 0], label=SIG_TYPE)
#     axes[0].plot(tt, amp[0, 0], label="Amplidue")
#     axes[0].legend()
#     # axes[0].set_xlabel("Time [s]")
#     axes[0].set_ylabel("Amplitude [?V]")
# 
#     axes[1].plot(tt, pha[0, 0], label="Phase")
#     axes[1].legend()
# 
#     axes[1].set_xlabel("Time [s]")
#     axes[1].set_ylabel("Phase [rad]")
# 
#     # plt.show()
#     mngs.io.save(fig, "traces.png")
# 
#     # Close
#     mngs.gen.close(CONFIG)
# 
# # EOF
# 
# """
# /home/ywatanabe/proj/entrance/mngs/dsp/_hilbert.py
# """
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/dsp/_hilbert.py
# --------------------------------------------------------------------------------
