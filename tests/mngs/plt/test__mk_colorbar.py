#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:16:21 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__mk_colorbar.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__mk_colorbar.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
def test_mk_colorbar():
    from mngs.plt._mk_colorbar import mk_colorbar

    # Test with default colors
    fig = mk_colorbar()

    # Check that it returns a figure
    assert isinstance(fig, plt.Figure)

    # Check with custom colors
    fig = mk_colorbar(start="red", end="green")
    assert isinstance(fig, plt.Figure)

    plt.close("all")


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt
    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_mk_colorbar.py
# --------------------------------------------------------------------------------
# import mngs
# import numpy as np
# import matplotlib
# # matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
#
#
# def mk_colorbar(start="white", end="blue"):
#     xx = np.linspace(0, 1, 256)
#
#     start = np.array(mngs.plt.colors.RGB_d[start])
#     end = np.array(mngs.plt.colors.RGB_d[end])
#     colors = (end-start)[:, np.newaxis]*xx
#
#     colors -= colors.min()
#     colors /= colors.max()
#
#     fig, ax = plt.subplots()
#     [ax.axvline(_xx, color=colors[:,i_xx]) for i_xx, _xx in enumerate(xx)]
#     ax.xaxis.set_ticks_position("none")
#     ax.yaxis.set_ticks_position("none")
#     ax.set_aspect(0.2)
#     return fig
#
#

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_mk_colorbar.py
# --------------------------------------------------------------------------------

# EOF