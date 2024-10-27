#!/usr/bin/env python3

try:
    from ._bonferroni_correction import bonferroni_correction
except ImportError as e:
    warnings.warn(f"Warning: Failed to import bonferroni_correction from ._bonferroni_correction.")

try:
    from ._brunner_munzel_test import brunner_munzel_test
except ImportError as e:
    warnings.warn(f"Warning: Failed to import brunner_munzel_test from ._brunner_munzel_test.")

try:
    from ._calc_partial_corr import calc_partial_corr
except ImportError as e:
    warnings.warn(f"Warning: Failed to import calc_partial_corr from ._calc_partial_corr.")

try:
    from ._corr_test import corr_test
except ImportError as e:
    warnings.warn(f"Warning: Failed to import corr_test from ._corr_test.")

try:
    from ._fdr_correction import fdr_correction
except ImportError as e:
    warnings.warn(f"Warning: Failed to import fdr_correction from ._fdr_correction.")

try:
    from ._general import kurtosis, mean, median, q, skewness, std, zscore
except ImportError as e:
    warnings.warn(f"Warning: Failed to import from ._general.")

try:
    from ._multicompair import multicompair
except ImportError as e:
    warnings.warn(f"Warning: Failed to import multicompair from ._multicompair.")

try:
    from ._nocorrelation_test import nocorrelation_test
except ImportError as e:
    warnings.warn(f"Warning: Failed to import nocorrelation_test from ._nocorrelation_test.")

try:
    from ._p2stars import p2stars
except ImportError as e:
    warnings.warn(f"Warning: Failed to import p2stars from ._p2stars.")

try:
    from ._smirnov_grubbs import smirnov_grubbs
except ImportError as e:
    warnings.warn(f"Warning: Failed to import smirnov_grubbs from ._smirnov_grubbs.")

try:
    from ._find_pval import find_pval
except ImportError as e:
    warnings.warn(f"Warning: Failed to import find_pval from ._find_pval.")


# from ._bonferroni_correction import bonferroni_correction
# from ._brunner_munzel_test import brunner_munzel_test
# from ._calc_partial_corr import calc_partial_corr
# from ._corr_test import corr_test
# from ._fdr_correction import fdr_correction#, fdr_correction_torch
# from ._general import kurtosis, mean, median, q, skewness, std, zscore
# from ._multicompair import multicompair
# from ._nocorrelation_test import nocorrelation_test
# from ._p2stars import p2stars
# from ._smirnov_grubbs import smirnov_grubbs
# from ._find_pval import find_pval
