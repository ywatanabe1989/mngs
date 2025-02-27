# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "ywatanabe (2024-11-02 23:09:49)"
# # File: ./mngs_repo/src/mngs/dsp/add_noise.py
# 
# import torch
# from ..decorators import torch_fn
# 
# def _uniform(shape, amp=1.0):
#     a, b = -amp, amp
#     return -amp + (2 * amp) * torch.rand(shape)
# 
# @torch_fn
# def gauss(x, amp=1.0):
#     noise = amp * torch.randn(x.shape)
#     return x + noise.to(x.device)
# 
# @torch_fn
# def white(x, amp=1.0):
#     return x + _uniform(x.shape, amp=amp).to(x.device)
# 
# @torch_fn
# def pink(x, amp=1.0, dim=-1):
#     """
#     Adds pink noise to a given tensor along a specified dimension.
# 
#     Parameters:
#     - x (torch.Tensor): The input tensor to which pink noise will be added.
#     - amp (float, optional): The amplitude of the pink noise. Defaults to 1.0.
#     - dim (int, optional): The dimension along which to add pink noise. Defaults to -1.
# 
#     Returns:
#     - torch.Tensor: The input tensor with added pink noise.
#     """
#     cols = x.size(dim)
#     noise = torch.randn(cols, dtype=x.dtype, device=x.device)
#     noise = torch.fft.rfft(noise)
#     indices = torch.arange(1, noise.size(0), dtype=x.dtype, device=x.device)
#     noise[1:] /= torch.sqrt(indices)
#     noise = torch.fft.irfft(noise, n=cols)
#     noise = noise - noise.mean()
#     noise_amp = torch.sqrt(torch.mean(noise**2))
#     noise = noise * (amp / noise_amp)
#     return x + noise.to(x.device)
# 
# @torch_fn
# def brown(x, amp=1.0, dim=-1):
#     noise = _uniform(x.shape, amp=amp)
#     noise = torch.cumsum(noise, dim=dim)
#     noise = mngs.dsp.norm.minmax(noise, amp=amp, dim=dim)
#     return x + noise.to(x.device)
# 
# if __name__ == "__main__":
#     import sys
# 
#     import matplotlib.pyplot as plt
#     import mngs
# 
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
# 
#     # Parameters
#     T_SEC = 1
#     FS = 128
# 
#     # Demo signal
#     xx, tt, fs = mngs.dsp.demo_sig(t_sec=T_SEC, fs=FS)
# 
#     funcs = {
#         "orig": lambda x: x,
#         "gauss": gauss,
#         "white": white,
#         "pink": pink,
#         "brown": brown,
#     }
# 
#     # Plots
#     fig, axes = mngs.plt.subplots(
#         nrows=len(funcs), ncols=2, sharex=True, sharey=True
#     )
#     count = 0
#     for (k, fn), axes_row in zip(funcs.items(), axes):
#         for ax in axes_row:
#             if count % 2 == 0:
#                 ax.plot(tt, fn(xx)[0, 0], label=k, c="blue")
#             else:
#                 ax.plot(tt, (fn(xx) - xx)[0, 0], label=f"{k} - orig", c="red")
#             count += 1
#             ax.legend(loc="upper right")
# 
#     fig.supxlabel("Time [s]")
#     fig.supylabel("Amplitude [?V]")
#     axes[0, 0].set_title("Signal + Noise")
#     axes[0, 1].set_title("Noise")
# 
#     mngs.io.save(fig, "traces.png")
# 
#     # Close
#     mngs.gen.close(CONFIG)
# 
# # EOF
# 
# """
# /home/ywatanabe/proj/entrance/mngs/dsp/add_noise.py
# """
# 
# # EOF

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
    sys.path.insert(0, os.path.join(project_root, "src"))

from ...src.mngs..dsp.add_noise import *

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
