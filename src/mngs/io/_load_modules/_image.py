#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-14 07:55:38 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_modules/_image.py

from typing import Any

from PIL import Image


def _load_image(lpath: str, **kwargs) -> Any:
    """Load image file."""
    if not any(
        lpath.endswith(ext) for ext in [".jpg", ".png", ".tiff", ".tif"]
    ):
        raise ValueError("Unsupported image format")
    return Image.open(lpath)


# EOF
