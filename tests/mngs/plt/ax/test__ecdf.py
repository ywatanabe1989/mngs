#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:30:34 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/test__ecdf.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/test__ecdf.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mngs.plt.ax._ecdf import ecdf

matplotlib.use("Agg")


class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        # Create sample data
        np.random.seed(42)
        self.data = [np.random.uniform(0, 1, 100)]

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)

    def test_basic_functionality(self):
        # Test basic ecdf functionality
        ax, df = ecdf(self.ax, self.data)

        # Check that lines were added to the plot
        assert len(self.ax.lines) > 0

        # Check that the returned DataFrame has the expected columns
        expected_columns = ["x", "y", "n", "x_step", "y_step"]
        assert all(col in df.columns for col in expected_columns)

        # Check that the ecdf values are between 0 and 100
        assert df["y"].min() >= 0
        assert df["y"].max() <= 100

        # Check that number of data points in df matches original data
        assert len(df) == len(self.data[0])
        assert df["n"].iloc[0] == len(self.data[0])

        # Check that axis limits were set
        assert self.ax.get_ylim() == (0, 100)
        assert self.ax.get_xlim() == (0, 1.0)

    def test_multiple_arrays(self):
        # Test with multiple arrays
        data = [np.random.uniform(0, 1, 50), np.random.uniform(0, 1, 30)]

        ax, df = ecdf(self.ax, data)

        # All data should be combined
        assert len(df) == 50 + 30
        assert df["n"].iloc[0] == 50 + 30

    def test_with_nan_values(self):
        # Test with data containing NaN values
        data_with_nan = [np.array([0.1, 0.2, np.nan, 0.4, 0.5, np.nan])]

        ax, df = ecdf(self.ax, data_with_nan)

        # NaN values should be removed
        assert len(df) == 4
        assert df["n"].iloc[0] == 4

    def test_with_plot_kwargs(self):
        # Test with additional plot kwargs
        ax, df = ecdf(self.ax, self.data, color="red", linewidth=2, alpha=0.5)

        # Plot style should be applied
        for line in self.ax.lines:
            if line.get_linestyle() != "None":
                assert line.get_color() == "red"
                assert line.get_linewidth() == 2
                assert line.get_alpha() == 0.5

    def test_step_values(self):
        # Test that the step values are correct
        sorted_data = np.sort(self.data[0])
        ax, df = ecdf(self.ax, self.data)

        # Check that x values in df match sorted data
        assert np.allclose(df["x"].values, sorted_data)

        # Check that ecdf values increase monotonically
        assert np.all(np.diff(df["y"].values) >= 0)

        # Check that the number of steps matches expectations
        expected_steps = 2 * len(sorted_data) - 1
        assert len(df["x_step"]) == expected_steps

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_ecdf.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-06 00:03:51 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/plt/ax/_ecdf.py
# 
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-07-05 20:06:17 (ywatanabe)"
# # ./src/mngs/plt/ax/_ecdf.py
# 
# import mngs
# import numpy as np
# import pandas as pd
# from ...pd._force_df import force_df
# 
# # def ecdf(ax, data):
# #     data = np.hstack(data)
# #     data = data[~np.isnan(data)]
# #     nn = len(data)
# 
# #     data_sorted = np.sort(data)
# #     ecdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
# 
# #     ax.step(data_sorted, ecdf, where="post")
# #     ax.plot(data_sorted, ecdf, marker=".", linestyle="none")
# 
# #     df = pd.DataFrame(
# #         {
# #             "x": data_sorted,
# #             "y": ecdf,
# #             "n": nn,
# #         }
# #     )
# #     return ax, df
# 
# 
# def ecdf(ax, data, **kwargs):
#     # Flatten and remove NaN values
#     data = np.hstack(data)
#     data = data[~np.isnan(data)]
#     nn = len(data)
# 
#     # Sort the data and compute the ECDF values
#     data_sorted = np.sort(data)
#     ecdf_perc = 100 * np.arange(1, len(data_sorted) + 1) / len(data_sorted)
# 
#     # Create the pseudo x-axis for step plotting
#     x_step = np.repeat(data_sorted, 2)[1:]
#     y_step = np.repeat(ecdf_perc, 2)[:-1]
# 
#     # Plot the ECDF_PERC using steps
#     ax.plot(x_step, y_step, drawstyle="steps-post", **kwargs)
# 
#     # Scatter the original data points
#     ax.plot(data_sorted, ecdf_perc, marker=".", linestyle="none")
# 
#     # Set ylim, xlim, and aspect ratio
#     ax.set_ylim(0, 100)  # Set y-axis limits
#     ax.set_xlim(0, 1.0)  # Set x-axis limits
#     # ax.set_aspect(1.0)  # Set aspect ratio
# 
#     # Create a DataFrame to hold the ECDF_PERC data
#     df = force_df(
#         {
#             "x": data_sorted,
#             "y": ecdf_perc,
#             "n": nn,
#             "x_step": x_step,
#             "y_step": y_step,
#         }
#     )
# 
#     # Return the matplotlib axis and DataFrame
#     return ax, df
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_ecdf.py
# --------------------------------------------------------------------------------
