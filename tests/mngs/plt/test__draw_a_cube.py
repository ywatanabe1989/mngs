#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-27 19:42:41 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__draw_a_cube.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__draw_a_cube.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Source code from: /home/ywatanabe/proj/mngs_dev/src/mngs/plt/_draw_a_cube.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
#
#
# def draw_a_cube(ax, r1, r2, r3, c="blue", alpha=1.0):
#     from itertools import combinations, product
#
#     for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
#         if np.sum(np.abs(s - e)) == r1[1] - r1[0]:
#             ax.plot3D(*zip(s, e), c=c, linewidth=3, alpha=alpha)
#         if np.sum(np.abs(s - e)) == r2[1] - r2[0]:
#             ax.plot3D(*zip(s, e), c=c, linewidth=3, alpha=alpha)
#         if np.sum(np.abs(s - e)) == r3[1] - r3[0]:
#             ax.plot3D(*zip(s, e), c=c, linewidth=3, alpha=alpha)
#     return ax

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Add source code to the top of Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs.plt._draw_a_cube import draw_a_cube


def test_draw_a_cube_creates_12_edges():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    r1 = [0, 1]
    r2 = [0, 1]
    r3 = [0, 1]
    draw_a_cube(ax, r1, r2, r3)
    # a cube has 12 edges
    assert len(ax.lines) == 12

# EOF