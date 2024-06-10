#!/usr/bin/env python3
# Time-stamp: "2024-06-10 23:17:49 (ywatanabe)"

from . import io, path
from ._sh import sh

_ = None  # to keep the importing order
from . import gen, general

_ = None
from . import ai, dsp, gists, linalg, ml, nn, os, pd, plt, stats, torch

_ = None
from . import res

__copyright__ = "Copyright (C) 2024 Yusuke Watanabe"
__version__ = "1.5.3"
__license__ = "MIT"
__author__ = "ywatanabe1989"
__author_email__ = "ywata1989@gmail.com"
__url__ = "https://github.com/ywatanabe1989/mngs"
