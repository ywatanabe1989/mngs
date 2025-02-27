#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-02-27 13:02:24 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/__init__.py

__file__ = "./src/mngs/__init__.py"
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-02-27 13:02:24 (ywatanabe)"
# File: ./mngs_repo/src/mngs/__init__.py

# os.getenv("MNGS_SENDER_GMAIL")
# os.getenv("MNGS_SENDER_GMAIL_PASSWORD")
# os.getenv("MNGS_RECIPIENT_GMAIL")
# os.getenv("MNGS_DIR", "/tmp/mngs/")

import os
import warnings
import asyncio

# Configure event loop before any async operations
# try:
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
# except Exception:
#     pass

# Configure warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

########################################
# Warnings
########################################

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
from . import parallel
from . import dev

# ########################################
# # Modules (python -m mngs print_config)
# ########################################
# from .gen._print_config import print_config
# # Usage: python -m mngs print_config

__copyright__ = "Copyright (C) 2024 Yusuke Watanabe"
__version__ = "1.10.1"
__license__ = "MIT"
__author__ = "ywatanabe1989"
__author_email__ = "ywatanabe@alumni.u-tokyo.ac.jp"
__url__ = "https://github.com/ywatanabe1989/mngs"

# EOF