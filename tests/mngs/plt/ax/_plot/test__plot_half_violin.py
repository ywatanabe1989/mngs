#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 21:54:40 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/_plot/test__plot_half_violin.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/_plot/test__plot_half_violin.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from mngs.plt.ax._plot._plot_half_violin import plot_half_violin

# class TestHalfViolin:

#     def setup_method(self):
#         # Setup test fixtures
#         self.fig = plt.figure()
#         self.ax = self.fig.add_subplot(111)

#         # Create sample data
#         np.random.seed(42)
#         data_size = 100
#         self.data = pd.DataFrame(
#             {
#                 "values": np.concatenate(
#                     [
#                         np.random.normal(0, 1, data_size),
#                         np.random.normal(3, 1, data_size),
#                     ]
#                 ),
#                 "group": np.repeat(["A", "B"], data_size),
#             }
#         )

#     def teardown_method(self):
#         # Clean up after tests
#         plt.close(self.fig)

#     def test_with_hue(self):
#         # Add a hue variable
#         self.data["subgroup"] = np.tile(["X", "Y"], len(self.data) // 2)

#         # Test with hue parameter
#         ax = plot_half_violin(
#             self.ax, data=self.data, x="group", y="values", hue="subgroup"
#         )

#         # Should have created a legend with entries for each subgroup
#         assert self.ax.get_legend() is not None

#     def test_plot_half_violin_savefig(self):
#         ax = plot_half_violin(self.ax, data=self.data, x="group", y="values")
#         ax.set_title("Half Violin Plot")

#         # Saving
#         from mngs.io import save

#         spath = f"./{os.path.basename(__file__)}.jpg"
#         save(self.fig, spath)


#         # Check saved file
#         ACTUAL_SAVE_DIR = __file__.replace(".py", "_out")
#         actual_spath = os.path.join(ACTUAL_SAVE_DIR, spath)
#         assert os.path.exists(
#             actual_spath
#         ), f"Failed to save figure to {spath}"
class TestPlotHalfViolin:
    def setup_method(self):
        # Setup test fixtures
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        # Create sample data
        np.random.seed(42)
        data_size = 100
        self.data = pd.DataFrame(
            {
                "values": np.concatenate(
                    [
                        np.random.normal(0, 1, data_size),
                        np.random.normal(3, 1, data_size),
                    ]
                ),
                "group": np.repeat(["A", "B"], data_size),
            }
        )
        # Create output directory if it doesn't exist
        self.out_dir = __file__.replace(".py", "_out")
        os.makedirs(self.out_dir, exist_ok=True)

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)

    def save_test_figure(self, method_name):
        """Helper method to save figure using method name"""
        from mngs.io import save

        spath = f"./{os.path.basename(__file__).replace('.py', '')}_{method_name}.jpg"
        save(self.fig, spath)
        # Check saved file
        actual_spath = os.path.join(self.out_dir, spath)
        assert os.path.exists(
            actual_spath
        ), f"Failed to save figure to {spath}"

    def test_with_hue(self):
        # Add a hue variable
        self.data["subgroup"] = np.tile(["X", "Y"], len(self.data) // 2)
        # Test with hue parameter
        ax = plot_half_violin(
            self.ax, data=self.data, x="group", y="values", hue="subgroup"
        )
        self.ax.set_title("Half Violin Plot with Hue")

        # Save figure
        self.save_test_figure("test_with_hue")

        # Should have created a legend with entries for each subgroup
        assert self.ax.get_legend() is not None

    def test_plot_half_violin_savefig(self):
        ax = plot_half_violin(self.ax, data=self.data, x="group", y="values")
        ax.set_title("Half Violin Plot")

        # Saving
        from mngs.io import save

        spath = f"./{os.path.basename(__file__)}.jpg"
        save(self.fig, spath)

        # Check saved file
        ACTUAL_SAVE_DIR = __file__.replace(".py", "_out")
        actual_spath = os.path.join(ACTUAL_SAVE_DIR, spath)
        assert os.path.exists(
            actual_spath
        ), f"Failed to save figure to {spath}"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_plot/_plot_half_violin.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-30 20:33:52 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_plot_violin_half.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/ax/_plot_violin_half.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# 
# def plot_half_violin(ax, data=None, x=None, y=None, hue=None, **kwargs):
# 
#     assert isinstance(
#         ax, matplotlib.axes._axes.Axes
#     ), "First argument must be a matplotlib axis"
# 
#     # Prepare data
#     df = data.copy()
#     if hue is None:
#         df["_hue"] = "default"
#         hue = "_hue"
# 
#     # Add fake hue for the right side
#     df["_fake_hue"] = df[hue] + "_right"
# 
#     # Adjust hue_order and palette if provided
#     if "hue_order" in kwargs:
#         kwargs["hue_order"] = kwargs["hue_order"] + [
#             h + "_right" for h in kwargs["hue_order"]
#         ]
# 
#     if "palette" in kwargs:
#         palette = kwargs["palette"]
#         if isinstance(palette, dict):
#             kwargs["palette"] = {
#                 **palette,
#                 **{k + "_right": v for k, v in palette.items()},
#             }
#         elif isinstance(palette, list):
#             kwargs["palette"] = palette + palette
# 
#     # Plot
#     sns.violinplot(
#         data=df, x=x, y=y, hue="_fake_hue", split=True, ax=ax, **kwargs
#     )
# 
#     # Remove right half of violins
#     for collection in ax.collections:
#         if isinstance(collection, plt.matplotlib.collections.PolyCollection):
#             collection.set_clip_path(None)
# 
#     # Adjust legend
#     if ax.legend_ is not None:
#         handles, labels = ax.get_legend_handles_labels()
#         ax.legend(handles[: len(handles) // 2], labels[: len(labels) // 2])
# 
#     return ax
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_plot/_plot_half_violin.py
# --------------------------------------------------------------------------------
