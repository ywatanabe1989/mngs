#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-30 17:05:54 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/__init__.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/__init__.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/__init__.py"

# os.getenv("MNGS_SENDER_GMAIL")
# os.getenv("MNGS_SENDER_GMAIL_PASSWORD")
# os.getenv("MNGS_RECIPIENT_GMAIL")
# os.getenv("MNGS_DIR", "/tmp/mngs/")

import warnings

# Configure warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

########################################
# Warnings
########################################

from . import types
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
from . import resource
from . import web
from . import db
from . import pd
from . import str
from . import parallel
from . import dt
from . import dev
from . import scholar

# from . import context


__version__ = "1.12.0"

# EOF
