#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:24:43 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__grid_image.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__grid_image.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import sys
from pathlib import Path

import mngs.plt._grid_image as gimod
import numpy as np

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from PIL import Image


def test_grid_image(monkeypatch):
    def dummy_load(path):
        return Image.new("RGB", (2, 3))

    monkeypatch.setattr(gimod.mngs.io, "load", dummy_load)
    paths = np.array([["a", None], [None, "b"]], dtype=object)
    img = gimod.grid_image(paths, default_color=(255, 0, 0))
    assert isinstance(img, Image.Image)
    # grid width = 2 cols * img width(2), height = 2 rows * img height(3)
    assert img.size == (4, 6)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_grid_image.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-05-13 09:01:29 (ywatanabe)"
#
#
# """
# This script does XYZ.
# """
#
#
# """
# Imports
# """
# from PIL import Image
#
#
# def grid_image(image_paths, default_color=(255, 255, 255)):
#     """
#     Create a grid of images from a 2D NumPy array of image paths.
#     Skips positions where image_paths is None.
#
#     Args:
#     image_paths (2D numpy array of str or None): Array of image file paths or None for empty slots
#     default_color (tuple): RGB color tuple for empty spaces
#
#     Returns:
#     PIL.Image: A new image consisting of the grid of images
#     """
#     import mngs
#
#     nrows, ncols = image_paths.shape
#
#     # Load images, skip None paths
#     images = []
#     for row in image_paths:
#         row_images = []
#         for path in row:
#             if path is not None:
#                 # img = Image.open(path)
#                 img = mngs.io.load(path)
#             else:
#                 img = None
#             row_images.append(img)
#         images.append(row_images)
#
#     # Assuming all images are the same size, use the first non-None image to determine size
#     for row in images:
#         for img in row:
#             if img is not None:
#                 img_width, img_height = img.size
#                 break
#         else:
#             continue
#         break
#     else:
#         raise ValueError("All image paths are None.")
#
#     # Create a new image with the total size
#     grid_width = img_width * ncols
#     grid_height = img_height * nrows
#     grid_image = Image.new("RGB", (grid_width, grid_height), default_color)
#
#     # Paste images into the grid
#     for y, row in enumerate(images):
#         for x, img in enumerate(row):
#             if img is not None:
#                 grid_image.paste(img, (x * img_width, y * img_height))
#
#     return grid_image

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_grid_image.py
# --------------------------------------------------------------------------------

# EOF