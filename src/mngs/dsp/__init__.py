#!/usr/bin/env python3

from ._gaussian_filter1d import gaussian_filter1d

# from ._load_BIDS import load_BIDS
from ._normalize_time import normalize_time
from ._take_random_references import take_random_references
from .bandpass_filters import bandpassfilter_np
from .demo_sig import demo_sig_np, demo_sig_torch
from .feature_extractors_torch import (
    amp_band_torch,
    amp_torch,
    bandpass_torch,
    bandstop_torch,
    beyond_r_sigma_ratio_torch,
    fft_torch,
    hilbert_torch,
    kurtosis_torch,
    mean_np_torch,
    median_torch,
    phase_band_torch,
    phase_torch,
    q25_torch,
    q75_torch,
    rfft_bands_torch,
    rfft_torch,
    rms_torch,
    skewness_torch,
    spectrogram_torch,
    std_np_torch,
    zscore_torch,
)
from .FeatureExtractorTorch import FeatureExtractorTorch
from .fft_amps import (
    calc_fft_amps_np_torch,
    calc_fft_powers_np_torch,
    calc_psd_np_torch,
)
from .PARAMS import BANDS_LIM_HZ_DICT
from .referencing import common_average
from .wavelet import wavelet
