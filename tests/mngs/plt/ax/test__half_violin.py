#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:33:45 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/test__half_violin.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/test__half_violin.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from mngs.plt.ax._half_violin import half_violin

class TestMainFunctionality:
 import pandas as pd
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

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)

    def test_basic_functionality(self):
        # Test with basic parameters
        ax = half_violin(self.ax, data=self.data, x="group", y="values")

        # Check that collections were added to the axis
        assert len(self.ax.collections) > 0

        # At least some collections should be PolyCollections (for the violins)
        poly_collections = [
            c
            for c in self.ax.collections
            if isinstance(c, matplotlib.collections.PolyCollection)
        ]
        assert len(poly_collections) > 0

        # Violin plot should have created legend entries for each group
        assert (
            self.ax.get_legend() is None
            or len(self.ax.get_legend().get_texts()) == 2
        )

    def test_with_hue(self):
        # Add a hue variable
        self.data["subgroup"] = np.tile(["X", "Y"], len(self.data) // 2)

        # Test with hue parameter
        ax = half_violin(
            self.ax, data=self.data, x="group", y="values", hue="subgroup"
        )

        # Should have created a legend with entries for each subgroup
        assert self.ax.get_legend() is not None
        legend_texts = [t.get_text() for t in self.ax.get_legend().get_texts()]
        assert "X" in legend_texts
        assert "Y" in legend_texts

    def test_with_custom_palette(self):
        # Test with custom color palette as list
        palette = ["red", "blue"]
        ax = half_violin(
            self.ax, data=self.data, x="group", y="values", palette=palette
        )

        # Get collection colors
        for collection in self.ax.collections:
            if isinstance(collection, matplotlib.collections.PolyCollection):
                colors = collection.get_facecolor()
                break

        # Colors should include red and blue
        red_values = matplotlib.colors.to_rgba("red")
        blue_values = matplotlib.colors.to_rgba("blue")

        # At least one collection should have these colors
        color_match = False
        for collection in self.ax.collections:
            if isinstance(collection, matplotlib.collections.PolyCollection):
                colors = collection.get_facecolor()
                if any(
                    np.allclose(c[:3], red_values[:3]) for c in colors
                ) or any(np.allclose(c[:3], blue_values[:3]) for c in colors):
                    color_match = True
                    break

        assert color_match

    def test_with_hue_order(self):
        # Add a hue variable
        self.data["subgroup"] = np.tile(["X", "Y"], len(self.data) // 2)

        # Test with hue_order parameter
        hue_order = ["Y", "X"]  # Reverse of default alphabetical order
        ax = half_violin(
            self.ax,
            data=self.data,
            x="group",
            y="values",
            hue="subgroup",
            hue_order=hue_order,
        )

        # Legend entries should follow the specified order
        legend_texts = [t.get_text() for t in self.ax.get_legend().get_texts()]
        assert legend_texts == hue_order

    def test_edge_cases(self):
        # Test with single category
        single_data = self.data[self.data["group"] == "A"].copy()
        ax = half_violin(self.ax, data=single_data, x="group", y="values")

        # Should still create violins
        assert (
            len(
                [
                    c
                    for c in self.ax.collections
                    if isinstance(c, matplotlib.collections.PolyCollection)
                ]
            )
            > 0
        )

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_half_violin.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-04 17:24:03 (ywatanabe)"
# # /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/plt/ax/_half_violin.py
# 
# """
# 1. Functionality:
#    - (e.g., Executes XYZ operation)
# 2. Input:
#    - (e.g., Required data for XYZ)
# 3. Output:
#    - (e.g., Results of XYZ operation)
# 4. Prerequisites:
#    - (e.g., Necessary dependencies for XYZ)
# 
# (Remove me: Please fill docstrings above, while keeping the bulette point style, and remove this instruction line)
# """
# 
# 
# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# def half_violin(ax, data=None, x=None, y=None, hue=None, **kwargs):
#     # Prepare data
#     df = data.copy()
#     if hue is None:
#         df['_hue'] = 'default'
#         hue = '_hue'
# 
#     # Add fake hue for the right side
#     df['_fake_hue'] = df[hue] + '_right'
# 
#     # Adjust hue_order and palette if provided
#     if 'hue_order' in kwargs:
#         kwargs['hue_order'] = kwargs['hue_order'] + [h + '_right' for h in kwargs['hue_order']]
# 
#     if 'palette' in kwargs:
#         palette = kwargs['palette']
#         if isinstance(palette, dict):
#             kwargs['palette'] = {**palette, **{k + '_right': v for k, v in palette.items()}}
#         elif isinstance(palette, list):
#             kwargs['palette'] = palette + palette
# 
#     # Plot
#     sns.violinplot(data=df, x=x, y=y, hue='_fake_hue', split=True, ax=ax, **kwargs)
# 
#     # Remove right half of violins
#     for collection in ax.collections:
#         if isinstance(collection, plt.matplotlib.collections.PolyCollection):
#             collection.set_clip_path(None)
# 
#     # Adjust legend
#     if ax.legend_ is not None:
#         handles, labels = ax.get_legend_handles_labels()
#         ax.legend(handles[:len(handles)//2], labels[:len(labels)//2])
# 
#     return ax

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_half_violin.py
# --------------------------------------------------------------------------------
