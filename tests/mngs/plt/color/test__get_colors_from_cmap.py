#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-02 18:00:18 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/color/test__get_colors_from_cmap.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/color/test__get_colors_from_cmap.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import numpy as np
import pytest


def test_get_color_from_cmap():
    from mngs.plt.color._get_colors_from_cmap import get_color_from_cmap

    # Test with default parameters
    color = get_color_from_cmap("viridis", 0.5)
    assert isinstance(color, tuple)
    assert len(color) == 4  # RGBA
    assert all(0 <= val <= 1 for val in color)  # Values within range

    # Test with custom value range
    color_range = get_color_from_cmap("plasma", 10, value_range=(0, 20))
    assert isinstance(color_range, tuple)
    assert len(color_range) == 4

    # Test with custom alpha
    color_alpha = get_color_from_cmap("Blues", 0.7, alpha=0.5)
    assert color_alpha[3] == 0.5  # Alpha should be set correctly

    # Test value outside range gets clipped
    color_clipped = get_color_from_cmap("coolwarm", 2.0, value_range=(0, 1))
    color_expected = get_color_from_cmap("coolwarm", 1.0, value_range=(0, 1))
    assert color_clipped == color_expected


def test_get_colors_from_cmap():
    from mngs.plt.color._get_colors_from_cmap import get_colors_from_cmap

    # Test with default parameters
    colors = get_colors_from_cmap("viridis", 5)
    assert isinstance(colors, list)
    assert len(colors) == 5
    assert all(isinstance(color, tuple) for color in colors)
    assert all(len(color) == 4 for color in colors)  # Each color is RGBA

    # Test with custom value range
    colors_range = get_colors_from_cmap("plasma", 3, value_range=(-1, 1))
    assert len(colors_range) == 3

    # Test colors are evenly spaced
    colors_check = get_colors_from_cmap("viridis", 3)
    cmap = matplotlib.cm.get_cmap("viridis")
    expected_colors = [
        tuple(list(cmap(0))[:3]) + (1.0,),
        tuple(list(cmap(0.5))[:3]) + (1.0,),
        tuple(list(cmap(1))[:3]) + (1.0,),
    ]

    for idx, color in enumerate(colors_check):
        assert color[:3] == pytest.approx(expected_colors[idx][:3], abs=1e-6)


def test_get_categorical_colors_from_cmap():
    from mngs.plt.color._get_colors_from_cmap import \
        get_categorical_colors_from_cmap

    # Test with list categories
    categories = ["cat", "dog", "bird"]
    cat_colors = get_categorical_colors_from_cmap("tab10", categories)

    assert isinstance(cat_colors, dict)
    assert set(cat_colors.keys()) == set(categories)
    assert all(len(color) == 4 for color in cat_colors.values())

    # Test with numpy array categories
    num_categories = np.array([1, 2, 3, 1, 2])
    num_cat_colors = get_categorical_colors_from_cmap(
        "viridis", num_categories, alpha=0.7
    )

    assert set(num_cat_colors.keys()) == {1, 2, 3}
    assert all(color[3] == 0.7 for color in num_cat_colors.values())

    # Test with duplicate categories
    duplicate_categories = ["A", "B", "A", "C", "B"]
    dup_cat_colors = get_categorical_colors_from_cmap(
        "plasma", duplicate_categories
    )

    assert set(dup_cat_colors.keys()) == {"A", "B", "C"}
    assert len(dup_cat_colors) == 3

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/color/_get_colors_from_cmap.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-02 17:53:27 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/color/_get_from_cmap.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/color/_get_from_cmap.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# from typing import List, Optional, Tuple, Union
# 
# import matplotlib
# import numpy as np
# 
# # class ColorGetter:
# #     # https://stackoverflow.com/questions/26108436/how-can-i-get-the-matplotlib-rgb-color-given-the-colormap-name-boundrynorm-an
# #     def __init__(self, cmap_name, start_val, stop_val):
# #         self.cmap_name = cmap_name
# #         self.cmap = plt.get_cmap(cmap_name)
# #         self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
# #         self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
# 
# #     def get_rgb(self, val):
# #         return self.scalarMap.to_rgba(val)
# 
# 
# def get_color_from_cmap(
#     cmap_name: str,
#     value: float,
#     value_range: Optional[Tuple[float, float]] = None,
#     alpha: float = 1.0,
# ) -> Tuple[float, float, float, float]:
#     """Get a color from a colormap at a specific value.
# 
#     Parameters
#     ----------
#     cmap_name : str
#         Name of the colormap (e.g., 'viridis', 'plasma', 'Blues')
#     value : float
#         Value to map to a color in the colormap
#     value_range : tuple of (float, float), optional
#         Range of values to map to the colormap. If None, uses (0, 1)
#     alpha : float, optional
#         Alpha value for the color (0.0 to 1.0), by default 1.0
# 
#     Returns
#     -------
#     tuple
#         RGBA color tuple with values from 0 to 1
#     """
#     # Get the colormap
#     cmap = matplotlib.cm.get_cmap(cmap_name)
# 
#     # Normalize the value
#     if value_range is None:
#         norm_value = value
#     else:
#         min_val, max_val = value_range
#         norm_value = (value - min_val) / (max_val - min_val)
# 
#     # Clip to ensure within range
#     norm_value = np.clip(norm_value, 0.0, 1.0)
# 
#     # Get the color
#     rgba_color = list(cmap(norm_value))
# 
#     # Set alpha
#     rgba_color[3] = alpha
# 
#     return tuple(rgba_color)
# 
# 
# def get_colors_from_cmap(
#     cmap_name: str,
#     n_colors: int,
#     value_range: Optional[Tuple[float, float]] = None,
#     alpha: float = 1.0,
# ) -> List[Tuple[float, float, float, float]]:
#     """Get a list of evenly spaced colors from a colormap.
# 
#     Parameters
#     ----------
#     cmap_name : str
#         Name of the colormap (e.g., 'viridis', 'plasma', 'Blues')
#     n_colors : int
#         Number of colors to sample from the colormap
#     value_range : tuple of (float, float), optional
#         Range of values to map to the colormap. If None, uses (0, 1)
#     alpha : float, optional
#         Alpha value for the colors (0.0 to 1.0), by default 1.0
# 
#     Returns
#     -------
#     list
#         List of RGBA color tuples with values from 0 to 1
#     """
#     if value_range is None:
#         values = np.linspace(0, 1, n_colors)
#     else:
#         values = np.linspace(value_range[0], value_range[1], n_colors)
# 
#     return [
#         get_color_from_cmap(cmap_name, val, value_range, alpha)
#         for val in values
#     ]
# 
# 
# def get_categorical_colors_from_cmap(
#     cmap_name: str, categories: Union[List, np.ndarray], alpha: float = 1.0
# ) -> dict:
#     """Map categorical values to colors from a colormap.
# 
#     Parameters
#     ----------
#     cmap_name : str
#         Name of the colormap (e.g., 'viridis', 'plasma', 'Blues')
#     categories : list or np.ndarray
#         List of categories to map to colors
#     alpha : float, optional
#         Alpha value for the colors (0.0 to 1.0), by default 1.0
# 
#     Returns
#     -------
#     dict
#         Dictionary mapping categories to RGBA color tuples
#     """
#     unique_categories = np.unique(categories)
#     n_categories = len(unique_categories)
# 
#     colors = get_colors_from_cmap(cmap_name, n_categories, alpha=alpha)
# 
#     return {cat: colors[idx] for idx, cat in enumerate(unique_categories)}
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/color/_get_colors_from_cmap.py
# --------------------------------------------------------------------------------
