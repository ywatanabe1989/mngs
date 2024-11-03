# src from here --------------------------------------------------------------------------------
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
    sys.path.insert(0, project_root)

from src.mngs.plt._grid_image import *

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
