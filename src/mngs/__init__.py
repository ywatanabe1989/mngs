#!/usr/bin/env python3
# Time-stamp: "2024-04-13 02:56:30 (ywatanabe)"

from . import io

_ = None  # to keep the importing order
from . import general

gen = general

_ = None  # to keep the importing order
from . import dsp, gists, linalg, ml, nn, os, plt, resource, stats, torch

# _ = None  # to keep the importing order

__copyright__ = "Copyright (C) 2024 Yusuke Watanabe"
__version__ = "1.3.0"
__license__ = "MIT"
__author__ = "ywatanabe1989"
__author_email__ = "ywata1989@gmail.com"
__url__ = "https://github.com/ywatanabe1989/mngs"
