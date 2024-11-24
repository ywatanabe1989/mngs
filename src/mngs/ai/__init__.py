#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-20 10:53:14 (ywatanabe)"
# File: ./mngs_repo/src/mngs/ai/__init__.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/ai/__init__.py"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "ywatanabe (2024-11-07 19:44:34)"
# File: ./mngs_repo/src/mngs/ai/__init__.py

import importlib
import inspect
import os

from ._gen_ai._genai_factory import genai_factory as GenAI

# Get the current directory
current_dir = os.path.dirname(__file__)

# Iterate through all Python files in the current directory
for filename in os.listdir(current_dir):
    if filename.endswith(".py") and not filename.startswith("__"):
        module_name = filename[:-3]  # Remove .py extension
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Import only functions and classes from the module
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if not name.startswith("_"):
                    globals()[name] = obj

# Clean up temporary variables
del os, importlib, inspect, current_dir, filename, module_name, module, name, obj

from . import metrics
from . import act
from . import clustering
from . import layer
from . import loss
from . import optim
from . import plt
from . import sk
from . import utils
from . import feature_extraction


# EOF
