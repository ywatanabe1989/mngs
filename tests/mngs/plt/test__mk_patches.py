#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 19:42:54 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__mk_patches.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__mk_patches.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/_mk_patches.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Time-stamp: "2021-11-27 18:45:23 (ylab)"
#
# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
#
#
# def mk_patches(colors, labels):
#     """
#     colors = ["red", "blue"]
#     labels = ["label_1", "label_2"]
#     ax.legend(handles=mngs.plt.mk_patches(colors, labels))
#     """
#
#     patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
#     return patches

import sys
from pathlib import Path

from mngs.plt._mk_patches import mk_patches

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))


import matplotlib.patches as mpatches


def test_mk_patches_basic():
    colors = ["#f00", "#0f0"]
    labels = ["a", "b"]
    patches = mk_patches(colors, labels)
    assert isinstance(patches, list)
    assert isinstance(patches[0], mpatches.Patch)
    assert patches[0].get_label() == "a"

# EOF