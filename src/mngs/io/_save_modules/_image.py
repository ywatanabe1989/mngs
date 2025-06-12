#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-12 13:45:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/_image.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_modules/_image.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Image file saving functionality for mngs.io.save

This module acts as a wrapper around the existing _save_image.py functionality
to maintain consistency with the modular save system.
"""

from ._save_image import _save_image


def save_image(obj, spath, **kwargs):
    """Handle image file saving.
    
    Parameters
    ----------
    obj : matplotlib.figure.Figure, matplotlib.axes.Axes, PIL.Image, or array-like
        Image object to save. Can be a matplotlib figure/axes, PIL Image, or numpy array
    spath : str
        Path where image file will be saved
    **kwargs
        Additional keyword arguments passed to the underlying save function
        
    Notes
    -----
    This is a wrapper around the existing _save_image function
    Supports formats: PNG, JPG, JPEG, GIF, TIFF, TIF, SVG
    """
    _save_image(obj, spath, **kwargs)


# EOF
=======
# Timestamp: "2025-05-18 14:52:34 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/io/_save_modules/_save_image.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/io/_save_modules/_save_image.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import io as _io

import plotly
from PIL import Image


def _save_image(obj, spath, **kwargs):
    # png
    if spath.endswith(".png"):
        # plotly
        if isinstance(obj, plotly.graph_objs.Figure):
            obj.write_image(file=spath, format="png")
        # PIL image
        elif isinstance(obj, Image.Image):
            obj.save(spath)
        # matplotlib
        else:
            try:
                obj.savefig(spath)
            except:
                obj.figure.savefig(spath)
        del obj

    # tiff
    elif spath.endswith(".tiff") or spath.endswith(".tif"):
        # PIL image
        if isinstance(obj, Image.Image):
            obj.save(spath)
        # matplotlib
        else:
            try:
                obj.savefig(spath, dpi=300, format="tiff")
            except:
                obj.figure.savefig(spath, dpi=300, format="tiff")

        del obj

    # jpeg
    elif spath.endswith(".jpeg") or spath.endswith(".jpg"):
        buf = _io.BytesIO()

        # plotly
        if isinstance(obj, plotly.graph_objs.Figure):
            obj.write_image(buf, format="png")
            buf.seek(0)
            img = Image.open(buf)
            img.convert("RGB").save(spath, "JPEG")
            buf.close()

        # PIL image
        elif isinstance(obj, Image.Image):
            obj.save(spath)

        # matplotlib
        else:
            try:
                obj.savefig(buf, format="png")
            except:
                obj.figure.savefig(buf, format="png")

            buf.seek(0)
            img = Image.open(buf)
            img.convert("RGB").save(spath, "JPEG")
            buf.close()
        del obj

    # SVG
    elif spath.endswith(".svg"):
        # Plotly
        if isinstance(obj, plotly.graph_objs.Figure):
            obj.write_image(file=spath, format="svg")
        # Matplotlib
        else:
            try:
                obj.savefig(spath, format="svg")
            except AttributeError:
                obj.figure.savefig(spath, format="svg")
        del obj

# EOF
>>>>>>> origin/main
