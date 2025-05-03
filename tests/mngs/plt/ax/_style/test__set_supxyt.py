#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 12:36:22 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/_style/test__set_supxyt.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/_style/test__set_supxyt.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_style/_set_supxyt.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-07-13 07:56:46 (ywatanabe)"
# # Author: Yusuke Watanabe (ywata1989@gmail.com)
#
# """
# This script does XYZ.
# """
#
# # Imports
# import matplotlib.pyplot as plt
#
# from ._format_label import format_label
#
#
# # Functions
# def set_supxyt(
#     ax, xlabel=False, ylabel=False, title=False, format_labels=True
# ):
#     """Sets xlabel, ylabel and title"""
#     fig = ax.get_figure()
#
#     # if xlabel is not False:
#     #     fig.supxlabel(xlabel)
#
#     # if ylabel is not False:
#     #     fig.supylabel(ylabel)
#
#     # if title is not False:
#     #     fig.suptitle(title)
#     if xlabel is not False:
#         xlabel = format_label(xlabel) if format_labels else xlabel
#         fig.supxlabel(xlabel)
#
#     if ylabel is not False:
#         ylabel = format_label(ylabel) if format_labels else ylabel
#         fig.supylabel(ylabel)
#
#     if title is not False:
#         title = format_label(title) if format_labels else title
#         fig.suptitle(title)
#
#     return ax
#
#
# if __name__ == "__main__":
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
#
#     # (YOUR AWESOME CODE)
#
#     # Close
#     mngs.gen.close(CONFIG)
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_style/_set_supxyt.py
# --------------------------------------------------------------------------------

# EOF