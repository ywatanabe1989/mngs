#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:20:56 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__draw_a_cube.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__draw_a_cube.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_draw_a_cube.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-27 20:03:25 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_draw_a_cube.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_draw_a_cube.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# import numpy as np
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
#
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_draw_a_cube.py
# --------------------------------------------------------------------------------

# EOF