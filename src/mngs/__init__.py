#!/usr/bin/env python3
# Time-stamp: "2024-10-12 12:16:51 (ywatanabe)"

# from . import io, path

# # Core modules
try:
    # from . import gen, general
    from ._sh import sh
except ImportError as e:
    print(f"Warning: Failed to import some core modules. Error: {e}")

# Additional modules
additional_modules = ['io', 'path', 'gen', 'general', 'ai', 'dsp', 'gists', 'linalg', 'ml', 'nn', 'os', 'pd', 'plt', 'stats', 'torch', 'tex', 'typing']
for module in additional_modules:
    try:
        exec(f"from . import {module}")
    except ImportError as e:
        print(f"Warning: Failed to import {module}. Error: {e}")

try:
    from . import res
except ImportError as e:
    print(f"Warning: Failed to import res. Error: {e}")

try:
    from . import web
except ImportError as e:
    print(f"Warning: Failed to import web. Error: {e}")

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
