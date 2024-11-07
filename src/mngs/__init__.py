#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-07 17:25:00 (ywatanabe)"
# File: ./mngs_repo/src/mngs/__init__.py


import warnings
from ._sh import sh
from . import io
from . import path
from . import dict
from . import gen
from . import decorators
from . import ai
from . import dsp
from . import gists
from . import linalg
from . import nn
from . import os
from . import plt
from . import stats
from . import torch
from . import tex
from . import types
from . import resource
from . import web
from . import db
from . import pd
from . import str
from . import dev

########################################
# MNGS variables
########################################
# sender_gmail = os.getenv("MNGS_SENDER_GMAIL")
# sender_password = os.getenv("MNGS_SENDER_GMAIL_PASSWORD")
# recipient_email = recipient_email or os.getenv("MNGS_RECIPIENT_GMAIL")
# os.getenv("MNGS_CACHE_DIR")


########################################
# Warnings
########################################
warnings.filterwarnings("ignore", message="There is no current event loop", category=DeprecationWarning)

########################################
# Try import
########################################
# import os
# from .gen._suppress_output import suppress_output
# _do_suppress = os.getenv("MNGS_SUPPRESS_IMPORTING_MESSAGES", "").lower() == "true"
# with suppress_output(suppress=__do_suppress):


########################################
# Core Modules
########################################

# ########################################
# # Modules (python -m mngs print_config)
# ########################################
# from .gen._print_config import print_config
# # Usage: python -m mngs print_config

__copyright__ = "Copyright (C) 2024 Yusuke Watanabe"
__version__ = "1.9.4"
__license__ = "MIT"
__author__ = "ywatanabe1989"
__author_email__ = "ywatanabe@alumni.u-tokyo.ac.jp"
__url__ = "https://github.com/ywatanabe1989/mngs"


# EOF
