#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:38:30 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/test__raster_plot.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/test__raster_plot.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mngs.plt.ax._raster_plot import raster_plot

matplotlib.use("Agg")  # Use non-GUI backend for testing


class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        # Sample positions data for testing
        self.positions = [
            [10, 50, 90],  # Channel 1 events
            [20, 60, 100],  # Channel 2 events
            [30, 70, 110],  # Channel 3 events
            [40, 80, 120],  # Channel 4 events
        ]

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)

    def test_basic_functionality(self):
        # Test basic raster plot creation
        ax, df = raster_plot(self.ax, self.positions)

        # Check that events were plotted (EventCollection objects added)
        assert len(self.ax.collections) == len(self.positions)

        # Check that df is a DataFrame with the right structure
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == len(self.positions)  # Column for each channel
        assert not df.empty

    def test_with_time_parameter(self):
        # Test with custom time parameter
        custom_time = np.linspace(0, 150, 200)
        ax, df = raster_plot(self.ax, self.positions, time=custom_time)

        # Check that time was used correctly
        assert df.index.equals(pd.Index(custom_time))
        assert len(df) == len(custom_time)

    def test_with_labels(self):
        # Test with channel labels
        labels = ["Channel A", "Channel B", "Channel C", "Channel D"]
        ax, df = raster_plot(self.ax, self.positions, labels=labels)

        # Check that a legend was created
        assert self.ax.get_legend() is not None

        # Check that the legend has the right number of entries
        handles, legend_labels = self.ax.get_legend_handles_labels()
        assert len(legend_labels) == len(labels)
        assert all(l1 == l2 for l1, l2 in zip(legend_labels, labels))

    def test_with_colors(self):
        # Test with custom colors
        colors = ["red", "green", "blue", "purple"]
        ax, df = raster_plot(self.ax, self.positions, colors=colors)

        # Check collections have the right colors
        assert len(self.ax.collections) == len(colors)
        for collection, color in zip(self.ax.collections, colors):
            assert collection.get_color()[0].tolist()[
                :3
            ] == matplotlib.colors.to_rgb(color)

    def test_with_mixed_types(self):
        # Test with mixed input types (single values and lists)
        mixed_positions = [
            10,  # Single value
            [20, 60],  # List
            30,  # Single value
            [40, 80],  # List
        ]

        ax, df = raster_plot(self.ax, mixed_positions)

        # Check that single values were properly handled
        assert len(self.ax.collections) == len(mixed_positions)

    def test_with_kwargs(self):
        # Test with additional kwargs
        ax, df = raster_plot(
            self.ax, self.positions, linewidths=2.0, linelengths=0.8
        )

        # Check that kwargs were applied
        for collection in self.ax.collections:
            assert collection.get_linewidths()[0] == 2.0
            assert collection.get_linelengths()[0] == 0.8

    def test_data_processing(self):
        # Check the DataFrame creation logic
        ax, df = raster_plot(self.ax, self.positions)

        # Verify that each channel has events marked at the right positions
        time_values = df.index.values

        # Find indices closest to our event positions
        for channel_idx, positions in enumerate(self.positions):
            for pos in positions:
                # Find rows in df where this channel has an event
                events = df[df.iloc[:, channel_idx] == channel_idx]

                # At least one event should be close to each position
                found = False
                for time in events.index:
                    if abs(time - pos) < 2.0:  # Allow some tolerance
                        found = True
                        break

                assert (
                    found
                ), f"No event found near position {pos} for channel {channel_idx}"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_raster_plot.py
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_raster_plot.py
# --------------------------------------------------------------------------------
