#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-02 18:29:20 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/_plot/test__plot_heatmap.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/_plot/test__plot_heatmap.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib.pyplot as plt
import numpy as np


def test_plot__heatmap_basic():
    fig, ax = plt.subplots()
    cm_arr = np.array([[0.8, 0.2], [0.3, 0.7]])
    ax = plot_heatmap(ax, cm_arr, labels=["Class A", "Class B"], cmap="YlGnBu")
    ax.set_title("Annotated Heatmap")

    # Saving
    from mngs.io import save

    spath = f"./basic.jpg"
    save(fig, spath)

    # Check saved file
    ACTUAL_SAVE_DIR = __file__.replace(".py", "_out")
    actual_spath = os.path.join(ACTUAL_SAVE_DIR, spath)
    assert os.path.exists(actual_spath), f"Failed to save figure to {spath}"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_plot/_plot_heatmap.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-02 18:34:08 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_plot/_plot_heatmap.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/ax/_plot/_plot_heatmap.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import pandas as pd
# import seaborn as sns
# 
# # def plot_heatmap(
# #     ax,
# #     conf_mat,
# #     labels=None,
# #     cmap="Blues",
# #     norm=None,
# #     annot=True,
# #     **kwargs,
# # ):
# #     conf_mat_df = pd.DataFrame(data=conf_mat)
# 
# #     if labels is not None:
# #         conf_mat_df.index = labels
# #         conf_mat_df.columns = labels
# 
# #     res = sns.heatmap(
# #         conf_mat_df,
# #         annot=annot,
# #         fmt=".3f",
# #         cmap=cmap,
# #         norm=norm,
# #         **kwargs,
# #     )  # cbar_kws={"shrink": 0.82}
# #     res.invert_yaxis()
# 
# #     # Remove spines
# #     [spine.set_visible(False) for _, spine in res.spines.items()]
# 
# #     return ax
# 
# 
# def plot_heatmap(
#     ax,
#     conf_mat,
#     labels=None,
#     cmap="Blues",
#     norm=None,
#     annot=True,
#     fmt=".3f",
#     invert_yaxis=True,
#     remove_spines=True,
#     **kwargs,
# ):
#     """
#     Plot a heatmap on the given axes.
# 
#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes
#         The axes to plot on
#     conf_mat : array-like
#         The confusion matrix or data to display as heatmap
#     labels : list, optional
#         Labels for both axes
#     cmap : str or matplotlib colormap, optional
#         Colormap to use, default "Blues"
#     norm : matplotlib.colors.Normalize, optional
#         Normalization for the colormap
#     annot : bool, optional
#         Whether to annotate cells with values, default True
#     fmt : str, optional
#         Format string for annotations, default ".3f"
#     invert_yaxis : bool, optional
#         Whether to invert the y-axis, default True
#     remove_spines : bool, optional
#         Whether to remove the spines, default True
#     **kwargs
#         Additional keyword arguments passed to sns.heatmap
# 
#     Returns
#     -------
#     matplotlib.axes.Axes
#         The axes object with the heatmap
#     """
#     # Convert to DataFrame for better labeling
#     conf_mat_df = pd.DataFrame(data=conf_mat)
# 
#     # Apply labels if provided
#     if labels is not None:
#         conf_mat_df.index = labels
#         conf_mat_df.columns = labels
# 
#     # Create heatmap
#     res = sns.heatmap(
#         conf_mat_df,
#         ax=ax,
#         annot=annot,
#         fmt=fmt,
#         cmap=cmap,
#         norm=norm,
#         **kwargs,
#     )
# 
#     # Invert y-axis if requested
#     if invert_yaxis:
#         res.invert_yaxis()
# 
#     # Remove spines if requested
#     if remove_spines:
#         for _, spine in res.spines.items():
#             spine.set_visible(False)
# 
#     # Return the axis for further customization
#     return ax
# 
# 
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from matplotlib import colors
# 
#     labels = ["T2T", "F2T", "T2F", "F2F"]
#     arr = np.random.randint(0, 10, len(labels) ** 2).reshape(
#         len(labels), len(labels)
#     )
# 
#     ## quantized, arbitrary range colormap you want
#     cmap = colors.ListedColormap(
#         ["navy", "royalblue", "lightsteelblue", "beige"],
#     )
#     norm = colors.BoundaryNorm([2, 4, 6, 8], cmap.N - 1)
# 
#     fig = plot_heatmap(arr, cmap=cmap, norm=norm, labels=labels)
#     fig.show()
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_plot/_plot_heatmap.py
# --------------------------------------------------------------------------------
