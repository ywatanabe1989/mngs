#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:25:48 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__get_RGBA_from_colormap.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__get_RGBA_from_colormap.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
from mngs.plt._get_RGBA_from_colormap import get_RGBA_from_colormap


def test_get_RGBA_from_colormap_range():
    rgba = get_RGBA_from_colormap(
        0.5, cmap="viridis", cmap_start_val=0, cmap_stop_val=1
    )
    arr = np.array(rgba)
    assert arr.shape == (4,)
    assert np.all(arr >= 0) and np.all(arr <= 1)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_get_RGBA_from_colormap.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
#
#
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib import cm
#
#
# class ColorGetter:
#     # https://stackoverflow.com/questions/26108436/how-can-i-get-the-matplotlib-rgb-color-given-the-colormap-name-boundrynorm-an
#     def __init__(self, cmap_name, start_val, stop_val):
#         self.cmap_name = cmap_name
#         self.cmap = plt.get_cmap(cmap_name)
#         self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
#         self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
#
#     def get_rgb(self, val):
#         return self.scalarMap.to_rgba(val)
#
#
# def get_RGBA_from_colormap(val, cmap="Blues", cmap_start_val=0, cmap_stop_val=1):
#     ColGetter = ColorGetter(cmap, cmap_start_val, cmap_stop_val)
#     return ColGetter.get_rgb(val)

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_get_RGBA_from_colormap.py
# --------------------------------------------------------------------------------

# EOF