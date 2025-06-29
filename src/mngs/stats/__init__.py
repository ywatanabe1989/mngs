#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-17 12:29:22 (ywatanabe)"
# File: ./mngs_repo/src/mngs/stats/__init__.py

THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/stats/__init__.py"

import os
import importlib
import inspect

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

from . import desc
from . import multiple
from . import tests
from ._describe_wrapper import describe
from ._nan_stats import nan, real
from ._corr_test_wrapper import corr_test, corr_test_spearman, corr_test_pearson
from .tests._corr_test import _compute_surrogate
from ._corr_test_multi import corr_test_multi, nocorrelation_test
from ._statistical_tests import brunner_munzel_test, smirnov_grubbs
from ._p2stars_wrapper import p2stars
from ._multiple_corrections import bonferroni_correction, fdr_correction, multicompair


# EOF
