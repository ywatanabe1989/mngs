#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:18:14 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__vizualize_colors.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__vizualize_colors.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing


def test_vizualize_colors():
    # Create a small test dictionary of colors
    test_colors = {"blue": [0, 0.5, 0.75, 0.9], "red": [1, 0.27, 0.2, 0.9]}

    # Patch plt.show to avoid displaying during tests
    original_show = plt.show
    plt.show = lambda: None

    try:
        # Test that the function runs without errors
        from mngs.plt.colors._vizualize_colors import vizualize_colors

        vizualize_colors(test_colors)

        # Additional assertions can be added here to verify plots
        plt.close("all")
    finally:
        # Restore original plt.show
        plt.show = original_show

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/color/_vizualize_colors.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-02 12:58:07 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/colors/_vizualize_colors.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/colors/_vizualize_colors.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import sys
# 
# import matplotlib.pyplot as plt
# import numpy as np
# 
# 
# def vizualize_colors(colors):
# 
#     def gen_rand_sample(size=100):
#         x = np.linspace(-1, 1, size)
#         y = np.random.normal(size=size)
#         s = np.random.randn(size)
#         return x, y, s
# 
#     from .. import subplots as mngs_plt_subplots
# 
#     fig, axes = mngs_plt_subplots(ncols=4)
# 
#     for ii, (color_str, rgba) in enumerate(colors.items()):
#         xx, yy, ss = gen_rand_sample()
# 
#         # Box color plot
#         axes[0].plot_rectangle(
#             xx=ii, yy=0, width=1, height=1, color=rgba, label=color_str
#         )
# 
#         # Line plot
#         axes[1].plot_shaded_line(
#             xx, yy - ss, yy, yy + ss, colors=rgba, label=color_str
#         )
# 
#         # Scatter plot
#         axes[2].scatter(xx, yy, color=rgba, label=color_str)
# 
#         # KDE plot
#         axes[3].plot_kde(yy, color=rgba, label=color_str)
# 
#     for ax in axes.flat:
#         # ax.axis("off")
#         ax.legend()
# 
#     plt.tight_layout()
#     plt.show()
# 
# 
# if __name__ == "__main__":
#     import mngs
# 
#     # # Argument Parser
#     # import argparse
#     # parser = argparse.ArgumentParser(description='')
#     # parser.add_argument('--var', '-v', type=int, default=1, help='')
#     # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
#     # args = parser.parse_args()
#     # Main
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False
#     )
#     main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/color/_vizualize_colors.py
# --------------------------------------------------------------------------------
