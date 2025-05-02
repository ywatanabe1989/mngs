#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 23:23:35 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__mk_colorbar.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__mk_colorbar.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib.pyplot as plt

def test_mk_colorbar():
    from mngs.plt.utils._mk_colorbar import mk_colorbar

    # Test with default colors
    fig = mk_colorbar()

    # Check that it returns a figure
    assert isinstance(fig, plt.Figure)

    # Check with custom colors
    fig = mk_colorbar(start="red", end="green")
    assert isinstance(fig, plt.Figure)

    plt.close("all")

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/utils/_mk_colorbar.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-01 23:37:12 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_mk_colorbar.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_mk_colorbar.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# # def mk_colorbar(start="white", end="blue"):
# #     xx = np.linspace(0, 1, 256)
# 
# #     start = np.array(mngs.plt.colors.RGB[start])
# #     end = np.array(mngs.plt.colors.RGB[end])
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
#     """Create a colorbar gradient between two colors.
# 
#     Args:
#         start (str): Starting color name
#         end (str): Ending color name
# 
#     Returns:
#         matplotlib.figure.Figure: Figure with colorbar
#     """
#     import matplotlib.colors as mcolors
#     import matplotlib.pyplot as plt
#     import numpy as np
#     # import mngs
#     from mngs.plt._PARAMS import RGB
# 
#     # Get RGB values for start and end colors
#     start_rgb = RGB[start]
#     end_rgb = RGB[end]
# 
#     # Create a colormap
#     colors = [start_rgb, end_rgb]
#     cmap = mcolors.LinearSegmentedColormap.from_list(
#         "custom_cmap", colors, N=256
#     )
# 
#     # Create the figure and plot the colorbar
#     fig, ax = plt.subplots(figsize=(6, 1))
#     gradient = np.linspace(0, 1, 256).reshape(1, -1)
#     ax.imshow(gradient, aspect="auto", cmap=cmap)
#     ax.set_xticks([])
#     ax.set_yticks([])
# 
#     return fig
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/utils/_mk_colorbar.py
# --------------------------------------------------------------------------------
