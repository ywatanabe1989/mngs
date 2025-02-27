# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 16:55:13 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_save_image.py
# 
# import io as _io
# from PIL import Image
# import plotly
# 
# def _save_image(obj, spath, **kwargs):
# 
#     # png
#     if spath.endswith(".png"):
#         # plotly
#         if isinstance(obj, plotly.graph_objs.Figure):
#             obj.write_image(file=spath, format="png")
#         # PIL image
#         elif isinstance(obj, Image.Image):
#             obj.save(spath)
#         # matplotlib
#         else:
#             try:
#                 obj.savefig(spath)
#             except:
#                 obj.figure.savefig(spath)
#         del obj
# 
#     # tiff
#     elif spath.endswith(".tiff") or spath.endswith(".tif"):
#         # PIL image
#         if isinstance(obj, Image.Image):
#             obj.save(spath)
#         # matplotlib
#         else:
#             try:
#                 obj.savefig(spath, dpi=300, format="tiff")
#             except:
#                 obj.figure.savefig(spath, dpi=300, format="tiff")
# 
#         del obj
# 
#     # jpeg
#     elif spath.endswith(".jpeg") or spath.endswith(".jpg"):
#         buf = _io.BytesIO()
# 
#         # plotly
#         if isinstance(obj, plotly.graph_objs.Figure):
#             obj.write_image(
#                 buf, format="png"
#             )
#             buf.seek(0)
#             img = Image.open(buf)
#             img.convert("RGB").save(spath, "JPEG")
#             buf.close()
# 
#         # PIL image
#         elif isinstance(obj, Image.Image):
#             obj.save(spath)
# 
#         # matplotlib
#         else:
#             try:
#                 obj.savefig(buf, format="png")
#             except:
#                 obj.figure.savefig(buf, format="png")
# 
#             buf.seek(0)
#             img = Image.open(buf)
#             img.convert("RGB").save(spath, "JPEG")
#             buf.close()
#         del obj
# 
#     # SVG
#     elif spath.endswith(".svg"):
#         # Plotly
#         if isinstance(obj, plotly.graph_objs.Figure):
#             obj.write_image(file=spath, format="svg")
#         # Matplotlib
#         else:
#             try:
#                 obj.savefig(spath, format="svg")
#             except AttributeError:
#                 obj.figure.savefig(spath, format="svg")
#         del obj
# 
# 
# # EOF

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
    sys.path.insert(0, os.path.join(project_root, "src"))

from ...src.mngs..io._save_image import *

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
