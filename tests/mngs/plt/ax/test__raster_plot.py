# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-13 12:57:26 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/plt/ax/_raster_plot.py
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-09-12 09:53:15 (ywatanabe)"
# # /home/ywatanabe/proj/mngs/src/mngs/plt/ax/_raster_plot.py
# 
# """This script provides a functionality of raster plotting"""
# 
# import sys
# from bisect import bisect_left
# 
# import matplotlib.pyplot as plt
# import mngs
# import numpy as np
# import pandas as pd
# 
# # def raster_plot(ax, positions, time=None, **kwargs):
# #     """
# #     Create a raster plot using eventplot and return the plot along with a DataFrame.
# 
# #     Parameters
# #     ----------
# #     ax : matplotlib.axes.Axes
# #         The axes on which to draw the raster plot.
# #     positions : list of lists or array-like
# #         Position of events for each channel. Each list corresponds to events of one channel.
# #     time : array-like, optional
# #         The time indices for the events. If None, time will be generated based on event positions.
# #     **kwargs : dict
# #         Additional keyword arguments to be passed to the eventplot function.
# 
# #     Returns
# #     -------
# #     ax : matplotlib.axes.Axes
# #         The axes with the raster plot.
# #     df : pandas.DataFrame
# #         A DataFrame where rows correspond to time indices and columns correspond to channels.
# #         Each cell contains the channel index for events at specific time indices.
# 
# #     Example
# #     -------
# #     positions = [[10, 50, 90], [20, 60, 100], [30, 70, 110]]
# #     fig, ax = plt.subplots()
# #     ax, df = raster_plot(ax, positions)
# #     plt.show()
# #     """
# 
# #     def ensure_list(positions):
# #         return [
# #             [pos] if isinstance(pos, (int, float)) else pos
# #             for pos in positions
# #         ]
# 
# #     def positions_to_df(positions, time):
# #         if time is None:
# #             time = np.linspace(
# #                 0, np.max([np.max(pos) for pos in positions]), 1000
# #             )
# 
# #         digi = np.full((len(positions), len(time)), np.nan, dtype=float)
# 
# #         for channel_index, channel_positions in enumerate(positions):
# #             for position in channel_positions:
# #                 insert_index = bisect_left(time, position)
# #                 if insert_index == len(time):
# #                     insert_index -= 1
# #                 digi[channel_index, insert_index] = channel_index
# 
# #         return pd.DataFrame(digi.T, index=time)
# 
# #     positions = ensure_list(positions)
# #     df = positions_to_df(positions, time)
# 
# #     ax.eventplot(positions, orientation="horizontal", **kwargs)
# #     return ax, df
# 
# 
# def raster_plot(ax, positions, time=None, labels=None, colors=None, **kwargs):
#     """
#     Create a raster plot using eventplot with custom labels and colors.
# 
#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes
#         The axes on which to draw the raster plot.
#     positions : list of lists or array-like
#         Position of events for each channel.
#     time : array-like, optional
#         The time indices for the events.
#     labels : list, optional
#         Labels for each channel.
#     colors : list, optional
#         Colors for each channel.
#     **kwargs : dict
#         Additional keyword arguments for eventplot.
# 
#     Returns
#     -------
#     ax : matplotlib.axes.Axes
#         The axes with the raster plot.
#     df : pandas.DataFrame
#         DataFrame with time indices and channel events.
#     """
# 
#     def ensure_list(positions):
#         return [
#             [pos] if isinstance(pos, (int, float)) else pos
#             for pos in positions
#         ]
# 
#     def positions_to_df(positions, time):
#         if time is None:
#             time = np.linspace(
#                 0, np.max([np.max(pos) for pos in positions]), 1000
#             )
# 
#         digi = np.full((len(positions), len(time)), np.nan, dtype=float)
# 
#         for channel_index, channel_positions in enumerate(positions):
#             for position in channel_positions:
#                 insert_index = bisect_left(time, position)
#                 if insert_index == len(time):
#                     insert_index -= 1
#                 digi[channel_index, insert_index] = channel_index
# 
#         return pd.DataFrame(digi.T, index=time)
# 
#     positions = ensure_list(positions)
#     df = positions_to_df(positions, time)
# 
#     # Handle colors and labels
#     if colors is None:
#         colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     if len(colors) < len(positions):
#         colors = colors * (len(positions) // len(colors) + 1)
# 
#     # Create event collection with colors
#     for i, (pos, color) in enumerate(zip(positions, colors)):
#         label = labels[i] if labels is not None and i < len(labels) else None
#         ax.eventplot(
#             pos, orientation="horizontal", colors=color, label=label, **kwargs
#         )
# 
#     if labels is not None:
#         ax.legend()
# 
#     return ax, df
# 
# 
# def test():
#     positions = [
#         [10, 50, 90],
#         [20, 60, 100],
#         [30, 70, 110],
#         [40, 80, 120],
#     ]
#     fig, ax = mngs.plt.subplots()
#     ax, df = raster_plot(ax, positions)
# 
# 
# if __name__ == "__main__":
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False
#     )
#     test()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)
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

from ...src.mngs..plt.ax._raster_plot import *

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
