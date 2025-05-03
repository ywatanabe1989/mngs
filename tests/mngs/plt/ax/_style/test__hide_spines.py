#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 12:36:11 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/_style/test__hide_spines.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/_style/test__hide_spines.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_style/_hide_spines.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-02 09:00:58 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_style/_hide_spines.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/ax/_style/_hide_spines.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# # Time-stamp: "2024-04-26 20:03:45 (ywatanabe)"
#
# import matplotlib
#
#
# def hide_spines(
#     axis,
#     top=True,
#     bottom=True,
#     left=True,
#     right=True,
#     ticks=True,
#     labels=True,
# ):
#     """
#     Hides the specified spines of a matplotlib Axes object and optionally removes the ticks and labels.
#
#     This function is designed to work with matplotlib Axes objects. It allows for a cleaner, more minimalist
#     presentation of plots by hiding the spines (the lines denoting the boundaries of the plot area) and optionally
#     removing the ticks and labels from the axes.
#
#     Arguments:
#         ax (matplotlib.axes.Axes): The Axes object for which the spines will be hidden.
#         top (bool, optional): If True, hides the top spine. Defaults to True.
#         bottom (bool, optional): If True, hides the bottom spine. Defaults to True.
#         left (bool, optional): If True, hides the left spine. Defaults to True.
#         right (bool, optional): If True, hides the right spine. Defaults to True.
#         ticks (bool, optional): If True, removes the ticks from the hidden spines' axes. Defaults to True.
#         labels (bool, optional): If True, removes the labels from the hidden spines' axes. Defaults to True.
#
#     Returns:
#         matplotlib.axes.Axes: The modified Axes object with the specified spines hidden.
#
#     Example:
#         >>> fig, ax = plt.subplots()
#         >>> hide_spines(ax, top=False, labels=False)
#         >>> plt.show()
#     """
#     assert isinstance(
#         axis, matplotlib.axes._axes.Axes
#     ), "First argument must be a matplotlib axis"
#
#     tgts = []
#     if top:
#         tgts.append("top")
#     if bottom:
#         tgts.append("bottom")
#     if left:
#         tgts.append("left")
#     if right:
#         tgts.append("right")
#
#     for tgt in tgts:
#         # Spines
#         axis.spines[tgt].set_visible(False)
#
#         # Ticks
#         if ticks:
#             if tgt == "bottom":
#                 axis.xaxis.set_ticks_position("none")
#             elif tgt == "left":
#                 axis.yaxis.set_ticks_position("none")
#
#         # Labels
#         if labels:
#             if tgt == "bottom":
#                 axis.set_xticklabels([])
#             elif tgt == "left":
#                 axis.set_yticklabels([])
#
#     return axis
#
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_style/_hide_spines.py
# --------------------------------------------------------------------------------

# EOF