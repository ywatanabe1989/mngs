#!/usr/bin/env python3
# Time-stamp: "2024-04-05 17:24:14 (ywatanabe)"


from . import general

gen = general  # alias

_ = None  # the order is the matter
from . import dsp, gists, io, linalg, ml, nn, os, plt, resource, stats, torch
from .general.debug import *

__copyright__ = "Copyright (C) 2024 Yusuke Watanabe"
__version__ = "1.2.2"
__license__ = "GPL3.0"
__author__ = "ywatanabe1989"
__author_email__ = "ywata1989@gmail.com"
__url__ = "https://github.com/ywatanabe1989/mngs"
