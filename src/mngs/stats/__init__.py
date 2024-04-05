#!/usr/bin/env python3

from ._bonferroni_correction import bonferroni_correction
from ._brunner_munzel_test import brunner_munzel_test
from ._calc_partial_corr import calc_partial_corr
from ._corr_test import corr_test
from ._fdr_correction import fdr_correction, fdr_correction_torch
from ._general import kurtosis, mean, median, q, skewness, std, zscore
from ._multicompair import multicompair
from ._nocorrelation_test import nocorrelation_test
from ._smirnov_grubbs import smirnov_grubbs
from ._to_asterisks import to_asterisks
