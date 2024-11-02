# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-02 21:28:58 (ywatanabe)"
# 
# """
# This script does XYZ.
# """
# 
# import sys
# 
# import matplotlib.pyplot as plt
# 
# import torch
# from ..decorators import torch_fn
# from mngs.nn import Hilbert
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
# 
# if __name__ == "__main__":
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

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.mngs.dsp/_hilbert.py import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass
    
    def teardown_method(self):
        # Clean up after tests
        pass
    
    def test_basic_functionality(self):
        # Basic test case
        pass
    
    def test_edge_cases(self):
        # Edge case testing
        pass
    
    def test_error_handling(self):
        # Error handling testing
        pass
