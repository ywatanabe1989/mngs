#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "ywatanabe (2024-11-03 00:02:48)"
# File: ./mngs_repo/src/mngs/stats/__init__.py
#!/usr/bin/env python3

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-22 19:51:47 (ywatanabe)"
# File: __init__.py

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

# EOF

# try:
#     from ._bonferroni_correction import bonferroni_correction
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import bonferroni_correction from ._bonferroni_correction.")

# try:
#     from ._brunner_munzel_test import brunner_munzel_test
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import brunner_munzel_test from ._brunner_munzel_test.")

# try:
#     from ._calc_partial_corr import calc_partial_corr
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import calc_partial_corr from ._calc_partial_corr.")

# try:
#     from ._corr_test import corr_test
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import corr_test from ._corr_test.")

# try:
#     from ._fdr_correction import fdr_correction
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import fdr_correction from ._fdr_correction.")

# try:
#     from ._gen import kurtosis, mean, median, q, skewness, std, zscore
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import from ._gen.")

# try:
#     from ._multicompair import multicompair
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import multicompair from ._multicompair.")

# try:
#     from ._nocorrelation_test import nocorrelation_test
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import nocorrelation_test from ._nocorrelation_test.")

# try:
#     from ._p2stars import p2stars
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import p2stars from ._p2stars.")

# try:
#     from ._smirnov_grubbs import smirnov_grubbs
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import smirnov_grubbs from ._smirnov_grubbs.")

# try:
#     from ._find_pval import find_pval
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import find_pval from ._find_pval.")


# from ._bonferroni_correction import bonferroni_correction
# from ._brunner_munzel_test import brunner_munzel_test
# from ._calc_partial_corr import calc_partial_corr
# from ._corr_test import corr_test
# from ._fdr_correction import fdr_correction#, fdr_correction_torch
# from ._gen import kurtosis, mean, median, q, skewness, std, zscore
# from ._multicompair import multicompair
# from ._nocorrelation_test import nocorrelation_test
# from ._p2stars import p2stars
# from ._smirnov_grubbs import smirnov_grubbs
# from ._find_pval import find_pval


# EOF
