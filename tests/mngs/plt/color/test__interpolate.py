#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 00:52:01 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/color/test__interpolate.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/color/test__interpolate.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib.colors as mcolors
import numpy as np


def test_interpolate():
    from mngs.plt.color._interpolate import interpolate

    # Test with basic colors and small number of points
    color_start = "red"
    color_end = "blue"
    num_points = 5

    colors = interpolate(color_start, color_end, num_points)

    # Check that we get the expected number of colors
    assert len(colors) == num_points

    # Check that each color is a list of 4 values (RGBA)
    for color in colors:
        assert len(color) == 4
        for value in color:
            assert 0 <= value <= 1

    # Check that the first and last colors match the inputs
    start_rgba = np.array(mcolors.to_rgba(color_start)).round(3)
    end_rgba = np.array(mcolors.to_rgba(color_end)).round(3)

    np.testing.assert_almost_equal(colors[0], start_rgba, decimal=3)
    np.testing.assert_almost_equal(colors[-1], end_rgba, decimal=3)


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/color/_interpolate.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-30 21:18:12 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/_interpolate.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_interpolate.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# import matplotlib.colors as mcolors
# import numpy as np
# from mngs.decorators import deprecated
#
#
# def gen_interpolate(color_start, color_end, num_points, round=3):
#     color_start_rgba = np.array(mcolors.to_rgba(color_start))
#     color_end_rgba = np.array(mcolors.to_rgba(color_end))
#     rgba_values = np.linspace(
#         color_start_rgba, color_end_rgba, num_points
#     ).round(round)
#     return [list(color) for color in rgba_values]
#
#
# @deprecated("Use gen_interpolate instead")
# def interpolate(color_start, color_end, num_points, round=3):
#     return gen_interpolate(color_start, color_end, num_points, round=round)
#
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/color/_interpolate.py
# --------------------------------------------------------------------------------

# EOF