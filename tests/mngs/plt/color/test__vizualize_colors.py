#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-02 23:16:42 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/color/test__vizualize_colors.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/color/test__vizualize_colors.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing


def test_vizualize_colors():
    from mngs.plt.color._vizualize_colors import vizualize_colors

    test_colors = {"blue": [0, 0.5, 0.75, 0.9], "red": [1, 0.27, 0.2, 0.9]}

    fig, ax = vizualize_colors(test_colors)

    from mngs.io import save

    spath = f"./{os.path.basename(__file__).replace('.py', '')}_test_vizualize_colors.jpg"
    save(fig, spath)
    # Check saved file
    out_dir = __file__.replace(".py", "_out")
    actual_spath = os.path.join(out_dir, spath)
    assert os.path.exists(actual_spath), f"Failed to save figure to {spath}"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/color/_vizualize_colors.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-03 00:53:43 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/color/_vizualize_colors.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/color/_vizualize_colors.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
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
#     fig, ax = mngs_plt_subplots()
# 
#     for ii, (color_str, rgba) in enumerate(colors.items()):
#         xx, yy, ss = gen_rand_sample()
# 
#         # # Box color plot
#         # ax.plot_rectangle(
#         #     xx=ii, yy=0, width=1, height=1, color=rgba, label=color_str
#         # )
# 
#         # Line plot
#         ax.plot_shaded_line(
#             xx, yy - ss, yy, yy + ss, color=rgba, label=color_str
#         )
# 
#         # # Scatter plot
#         # axes[2].scatter(xx, yy, color=rgba, label=color_str)
# 
#         # # KDE plot
#         # axes[3].plot_kde(yy, color=rgba, label=color_str)
# 
#     # for ax in axes.flat:
#     #     # ax.axis("off")
#     #     ax.legend()
# 
#     ax.legend()
#     # plt.tight_layout()
#     # plt.show()
#     return fig, ax
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/color/_vizualize_colors.py
# --------------------------------------------------------------------------------
