# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 16:57:29 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_save_mp4.py
# 
# from matplotlib import animation
# 
# def _mk_mp4(fig, spath_mp4):
#     axes = fig.get_axes()
# 
#     def init():
#         return (fig,)
# 
#     def animate(i):
#         for ax in axes:
#             ax.view_init(elev=10.0, azim=i)
#         return (fig,)
# 
#     anim = animation.FuncAnimation(
#         fig, animate, init_func=init, frames=360, interval=20, blit=True
#     )
# 
#     writermp4 = animation.FFMpegWriter(
#         fps=60, extra_args=["-vcodec", "libx264"]
#     )
#     anim.save(spath_mp4, writer=writermp4)
#     print("\nSaving to: {}\n".format(spath_mp4))
# 
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
    sys.path.insert(0, project_root)

from src.mngs.io/_save_mp4.py import *

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
