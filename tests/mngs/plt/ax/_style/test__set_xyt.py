#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 12:36:31 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/_style/test__set_xyt.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/_style/test__set_xyt.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_style/_set_xyt.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-07-13 08:14:19 (ywatanabe)"
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
# def set_xyt(ax, x=False, y=False, t=False, format_labels=True):
#     """Sets xlabel, ylabel and title"""
#
#     if x is not False:
#         x = format_label(x) if format_labels else x
#         ax.set_xlabel(x)
#
#     if y is not False:
#         y = format_label(y) if format_labels else y
#         ax.set_ylabel(y)
#
#     if t is not False:
#         t = format_label(t) if format_labels else t
#         ax.set_title(t)
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
#
# """
# /ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/plt/ax/_set_lt.py
# """

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_style/_set_xyt.py
# --------------------------------------------------------------------------------

# EOF