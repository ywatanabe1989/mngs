#!/usr/bin/env python3
# Time-stamp: "2024-04-08 12:29:52 (ywatanabe)"


# gen = general  # alias
from . import io

_ = None  # the order is the matter
from . import general

gen = general

_ = None  # the order is the matter
from . import dsp, gists, linalg, ml, nn, os, plt, resource, stats, torch

_ = None  # the order is the matter


# from .general.debug import *

__copyright__ = "Copyright (C) 2024 Yusuke Watanabe"
__version__ = "1.2.3"
__license__ = "MIT"
__author__ = "ywatanabe1989"
__author_email__ = "ywata1989@gmail.com"
__url__ = "https://github.com/ywatanabe1989/mngs"
