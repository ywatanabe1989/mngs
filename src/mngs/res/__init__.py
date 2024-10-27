#!/usr/bin/env python3

# import warnings

# try:
#     from ._get_proc_usages import get_proc_usages
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import get_proc_usages from ._get_proc_usages.")

# try:
#     from ._get_specs import get_specs
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import get_specs from ._get_specs.")

# try:
#     from ._rec_procs import rec_procs
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import rec_procs from ._rec_procs.")

# try:
#     from . import _utils
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import _utils.")


from ._get_proc_usages import get_proc_usages
from ._get_specs import get_specs
from ._rec_procs import rec_procs
from . import _utils

_ = None  # keep the importing orders

# from .limit_RAM import get_RAM, limit_RAM

# from .get import system_info

## EOF
