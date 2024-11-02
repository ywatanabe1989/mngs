#!/usr/bin/env python3
# Time-stamp: "2024-10-27 13:21:04 (ywatanabe)"

########################################
# Warnings
########################################
import warnings
warnings.filterwarnings("ignore", message="There is no current event loop", category=DeprecationWarning)

########################################
# Try import
########################################
import os
# from .gen._suppress_output import suppress_output
# _do_suppress = os.getenv("MNGS_SUPPRESS_IMPORTING_MESSAGES", "").lower() == "true"
# with suppress_output(suppress=__do_suppress):

########################################
# Core Modules
########################################
from ._sh import sh
from . import io
from . import path
from . import general
from . import gen
from . import ai
from . import ml
from . import dsp
from . import gists
from . import linalg
from . import nn
from . import os
from . import plt
from . import stats
from . import torch
from . import tex
from . import typing
from . import res
from . import web
from . import db
from . import pd

# ########################################
# # Lazy importing
# ########################################
# from importlib import import_module
# class LazyModule:
#     def __init__(self, name):
#         self._name = name
#         self._mod = None

#     def __getattr__(self, attr):
#         if self._mod is None:
#             self._mod = import_module(f'.{self._name}', 'mngs')
#         return getattr(self._mod, attr)

# # Replace direct imports with lazy ones
# io = LazyModule('io')
# path = LazyModule('path')
# general = LazyModule('general')
# gen = LazyModule('gen')
# ai = LazyModule('ai')
# ml = LazyModule('ml')
# dsp = LazyModule('dsp')
# gists = LazyModule('gists')
# linalg = LazyModule('linalg')
# nn = LazyModule('nn')
# os = LazyModule('os')
# plt = LazyModule('plt')
# stats = LazyModule('stats')
# torch = LazyModule('torch')
# tex = LazyModule('tex')
# typing = LazyModule('typing')
# res = LazyModule('res')
# web = LazyModule('web')
# db = LazyModule('db')
# pd = LazyModule('pd')


########################################
# Modules (python -m mngs print_config)
########################################
from .gen._print_config import print_config
# Usage: python -m mngs print_config

__copyright__ = "Copyright (C) 2024 Yusuke Watanabe"
__version__ = "1.8.1"
__license__ = "MIT"
__author__ = "ywatanabe1989"
__author_email__ = "ywatanabe@alumni.u-tokyo.ac.jp"
__url__ = "https://github.com/ywatanabe1989/mngs"
