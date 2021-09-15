#!/usr/bin/env python3
# Time-stamp: "2021-09-16 01:15:21 (ywatanabe)"

# from mngs import *
from . import dsp
from . import general
from . import ml
from . import plt
from . import stats


# from . import *
# from .dsp import *
# from .general import *
# from .ml import *
# from .plt import *
# from .stats import *
# from .io import *

__copyright__ = "Copyright (C) 2021 Yusuke Watanabe"
__version__ = "0.0.4"
__license__ = "GPL3.0"
__author__ = "ywatanabe1989"
__author_email__ = "ywata1989@gmail.com"
__url__ = "https://github.com/ywatanabe1989/mngs"

__all__ = [
    "dsp",
    "general",
    "ml",
    "plt",
    "stats",
]
# import os, glob

# __all__ = [
#     os.path.split(os.path.splitext(file)[0])[1]
#     for file in glob.glob(os.path.join(os.path.dirname(__file__), "[a-zA-Z0-9]*.py"))
# ]
