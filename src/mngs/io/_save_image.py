#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 16:55:13 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_save_image.py

import io as _io
from PIL import Image
import plotly

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
            obj.write_image(
                buf, format="png"
            )
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
