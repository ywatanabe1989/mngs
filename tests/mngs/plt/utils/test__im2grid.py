#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 23:19:54 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__im2grid.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__im2grid.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import mngs
import numpy as np
from mngs.plt.utils._im2grid import im2grid
from PIL import Image


def test_grid_image(monkeypatch):
    def dummy_load(path):
        return Image.new("RGB", (2, 3))

    monkeypatch.setattr(mngs.io, "load", dummy_load)
    paths = np.array([["a", None], [None, "b"]], dtype=object)
    # img = im2grid(paths, default_color=(255, 0, 0))
    img = im2grid(paths, default_color=(255, 0, 0))

    # Save the grid image
    from mngs.io import save

    spath = f"./{os.path.basename(__file__).replace('.py', '')}_test_grid_image.jpg"
    save(img, spath)

    # Check saved file
    ACTUAL_SAVE_DIR = __file__.replace(".py", "_out")
    actual_spath = os.path.join(ACTUAL_SAVE_DIR, spath)
    assert os.path.exists(
        actual_spath
    ), f"Failed to save grid image to {spath}"

    # Original assertions
    assert isinstance(img, Image.Image)
    # grid width = 2 cols * img width(2), height = 2 rows * img height(3)
    assert img.size == (4, 6)


def test_single_image(monkeypatch):
    def dummy_load(path):
        return Image.new("RGB", (4, 4))

    monkeypatch.setattr(mngs.io, "load", dummy_load)
    paths = np.array([["a"]], dtype=object)
    img = im2grid(paths)

    # Save the image
    from mngs.io import save

    spath = f"./{os.path.basename(__file__).replace('.py', '')}_test_single_image.jpg"
    save(img, spath)

    # Check saved file
    ACTUAL_SAVE_DIR = __file__.replace(".py", "_out")
    actual_spath = os.path.join(ACTUAL_SAVE_DIR, spath)
    assert os.path.exists(
        actual_spath
    ), f"Failed to save single image to {spath}"

    assert isinstance(img, Image.Image)
    assert img.size == (4, 4)


def test_custom_default_color(monkeypatch):
    def dummy_load(path):
        return Image.new("RGB", (3, 3))

    monkeypatch.setattr(mngs.io, "load", dummy_load)
    paths = np.array([["a", None], [None, "b"]], dtype=object)
    custom_color = (0, 255, 0)  # Green
    img = im2grid(paths, default_color=custom_color)

    # Save the image
    from mngs.io import save

    spath = f"./{os.path.basename(__file__).replace('.py', '')}_test_custom_default_color.jpg"
    save(img, spath)

    # Check saved file
    ACTUAL_SAVE_DIR = __file__.replace(".py", "_out")
    actual_spath = os.path.join(ACTUAL_SAVE_DIR, spath)
    assert os.path.exists(
        actual_spath
    ), f"Failed to save image with custom color to {spath}"

    assert isinstance(img, Image.Image)
    assert img.size == (6, 6)


# def test_grid_image(monkeypatch):
#     def dummy_load(path):
#         return Image.new("RGB", (2, 3))

#     monkeypatch.setattr(mngs.io, "load", dummy_load)
#     paths = np.array([["a", None], [None, "b"]], dtype=object)
#     img = im2grid(paths, default_color=(255, 0, 0))

#     # Save the grid image
#     out_dir = __file__.replace(".py", "_out")
#     os.makedirs(out_dir, exist_ok=True)

#     filename = (
#         os.path.basename(__file__).replace(".py", "") + "_test_grid_image.jpg"
#     )
#     save_path = os.path.join(out_dir, filename)

#     img.save(save_path)
#     assert os.path.exists(
#         save_path
#     ), f"Failed to save grid image to {save_path}"

#     # Original assertions
#     assert isinstance(img, Image.Image)
#     # grid width = 2 cols * img width(2), height = 2 rows * img height(3)
#     assert img.size == (4, 6)


# def test_single_image(monkeypatch):
#     def dummy_load(path):
#         return Image.new("RGB", (4, 4))

#     monkeypatch.setattr(mngs.io, "load", dummy_load)
#     paths = np.array([["a"]], dtype=object)
#     img = im2grid(paths)

#     # Save the image
#     out_dir = __file__.replace(".py", "_out")
#     os.makedirs(out_dir, exist_ok=True)

#     filename = os.path.basename(__file__).replace('.py', '') + "_test_single_image.jpg"
#     save_path = os.path.join(out_dir, filename)

#     img.save(save_path)
#     assert os.path.exists(save_path), f"Failed to save single image to {save_path}"

#     assert isinstance(img, Image.Image)
#     assert img.size == (4, 4)

# def test_custom_default_color(monkeypatch):
#     def dummy_load(path):
#         return Image.new("RGB", (3, 3))

#     monkeypatch.setattr(mngs.io, "load", dummy_load)
#     paths = np.array([["a", None], [None, "b"]], dtype=object)
#     custom_color = (0, 255, 0)  # Green
#     img = im2grid(paths, default_color=custom_color)

#     # Save the image
#     out_dir = __file__.replace(".py", "_out")
#     os.makedirs(out_dir, exist_ok=True)

#     filename = os.path.basename(__file__).replace('.py', '') + "_test_custom_default_color.jpg"
#     save_path = os.path.join(out_dir, filename)

#     img.save(save_path)
#     assert os.path.exists(save_path), f"Failed to save image with custom color to {save_path}"

#     assert isinstance(img, Image.Image)
#     assert img.size == (6, 6)

#     # To validate the background color, we'd need to check pixel values
#     # This depends on how PIL handles colors and may require more detailed testing

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/utils/_im2grid.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-01 23:21:22 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_im2grid.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_im2grid.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# from PIL import Image
# 
# 
# def im2grid(image_paths, default_color=(255, 255, 255)):
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
#     from mngs.io import load as mngs_io_load
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
#                 img = mngs_io_load(path)
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
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/utils/_im2grid.py
# --------------------------------------------------------------------------------
