#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 12:37:08 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/test__tpl.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/test__tpl.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import numpy as np
import pytest
import termplotlib as tpl

matplotlib.use("Agg")  # Use non-interactive backend for testing

from mngs.plt._tpl import termplot


class DummyFig:
    def __init__(self):
        self.plots = []

    def plot(self, x, y):
        self.plots.append((list(x), list(y)))

    def show(self):
        pass


def test_termplot_one_arg(monkeypatch):
    dummy = DummyFig()
    monkeypatch.setattr(tpl, "figure", lambda *args, **kwargs: dummy)
    y_values = np.array([0, 1, 2, 3])
    termplot(y_values)
    assert len(dummy.plots) == 1
    expected_x = list(np.arange(len(y_values)))
    expected_y = list(y_values)
    assert dummy.plots[0] == (expected_x, expected_y)


def test_termplot_two_args(monkeypatch):
    dummy = DummyFig()
    monkeypatch.setattr(tpl, "figure", lambda *args, **kwargs: dummy)
    x_values = np.array([10, 20, 30])
    y_values = np.array([1, 2, 3])
    termplot(x_values, y_values)
    assert dummy.plots[0] == (list(x_values), list(y_values))


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_tpl.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-03-31 11:58:28 (ywatanabe)"
#
# import numpy as np
# import termplotlib as tpl
#
#
# def termplot(*args):
#     """
#     Plots given y values against x using termplotlib, or plots a single y array against its indices if x is not provided.
#
#     Parameters:
#     - *args: Accepts either one argument (y values) or two arguments (x and y values).
#
#     Returns:
#     None. Displays the plot in the terminal.
#     """
#     if len(args) == 1:
#         y = args[0]  # [REVISED]
#         x = np.arange(len(y))
#
#     if len(args) == 2:
#         x, y = args
#
#     fig = tpl.figure()
#     fig.plot(x, y)
#     fig.show()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_tpl.py
# --------------------------------------------------------------------------------

# EOF