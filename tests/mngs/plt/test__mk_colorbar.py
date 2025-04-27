# Source code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_mk_colorbar.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-27 12:51:43 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_mk_colorbar.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_mk_colorbar.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import matplotlib.pyplot as plt
# import numpy as np
# 
# from ._PARAMS import PARAMS
# 
# # matplotlib.use("TkAgg")
# 
# # def mk_colorbar(start="white", end="blue"):
# #     xx = np.linspace(0, 1, 256)
# 
# #     start = np.array(mngs.plt.colors.RGB_d[start])
# #     end = np.array(mngs.plt.colors.RGB_d[end])
# #     colors = (end - start)[:, np.newaxis] * xx
# 
# #     colors -= colors.min()
# #     colors /= colors.max()
# 
# #     fig, ax = plt.subplots()
# #     [ax.axvline(_xx, color=colors[:, i_xx]) for i_xx, _xx in enumerate(xx)]
# #     ax.xaxis.set_ticks_position("none")
# #     ax.yaxis.set_ticks_position("none")
# #     ax.set_aspect(0.2)
# #     return fig
# 
# 
# def mk_colorbar(start="white", end="blue"):
#     xx = np.linspace(0, 1, 256)
#     rgb_start = np.array(PARAMS["RGB"][start])
#     rgb_end = np.array(PARAMS["RGB"][end])
#     # linear gradient in normalized space
#     grad = (rgb_end - rgb_start)[..., None] * xx[None, :]
#     grad = (rgb_start[..., None] + grad) / 255
#     fig, ax = plt.subplots()
#     for i, v in enumerate(xx):
#         ax.axvline(v, color=tuple(grad[:, i]))
#     ax.xaxis.set_ticks([])
#     ax.yaxis.set_ticks([])
#     ax.set_aspect(0.2)
#     return fig
# 
# # EOF
#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.plt._mk_colorbar import *

class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        raise NotImplementedError("Test not yet implemented")

    def test_edge_cases(self):
        # Edge case testing
        raise NotImplementedError("Test not yet implemented")

    def test_error_handling(self):
        # Error handling testing
        raise NotImplementedError("Test not yet implemented")

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
