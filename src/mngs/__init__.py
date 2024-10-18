#!/usr/bin/env python3
# Time-stamp: "2024-10-15 00:36:24 (ywatanabe)"

# from . import io, path
import sys, os

from .gen import suppress_output

suppress = os.getenv("MNGS_SUPPRESS_IMPORTING_MESSAGES", "").lower() == "true"

with suppress_output(suppress=suppress):
    # Core modules
    try:
        from ._sh import sh
    except ImportError as e:
        pass # print(f"Warning: Failed to import some core modules. Error: {e}")

    # Additional modules
    additional_modules = [
        "io",
        "path",
        "gen",
        "general",
        "ai",
        "dsp",
        "gists",
        "linalg",
        "ml",
        "nn",
        "os",
        "pd",
        "plt",
        "stats",
        "torch",
        "tex",
        "typing",
        "res",
        "web",
        "db",
    ]
    for module in additional_modules:
        try:
            exec(f"from . import {module}")
        except ImportError as e:
            pass # print(f"Warning: Failed to import {module}. Error: {e}")


    from ._sh import sh


# Modules
from .gen._print_config import print_config

__copyright__ = "Copyright (C) 2024 Yusuke Watanabe"
__version__ = "1.8.0"
__license__ = "MIT"
__author__ = "ywatanabe1989"
__author_email__ = "ywatanabe@alumni.u-tokyo.ac.jp"
__url__ = "https://github.com/ywatanabe1989/mngs"

# #!/usr/bin/env python3
# # Time-stamp: "2024-10-08 20:49:56 (ywatanabe)"


# from . import io, path

# None  # to keep the importing order
# from . import gen, general
# from ._sh import sh

# None
# from . import ai, dsp, gists, linalg, ml, nn, os, pd, plt, stats, torch, tex, typing

# None
# from . import res

# None  # to keep the importing order
# from . import web

# __copyright__ = "Copyright (C) 2024 Yusuke Watanabe"
# __version__ = "1.8.0"
# __license__ = "MIT"
# __author__ = "ywatanabe1989"
# __author_email__ = "ywatanabe@alumni.u-tokyo.ac.jp"
# __url__ = "https://github.com/ywatanabe1989/mngs"
