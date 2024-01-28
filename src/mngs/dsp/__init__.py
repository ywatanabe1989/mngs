#!/usr/bin/env python3

# from .wavelet import wavelet
from .demo_sig import demo_sig_np, demo_sig_torch
from .feature_extractors import (
    FeatureExtractorTorch,
    amp,
    amp_band,
    bandpass,
    bandstop,
    beyond_r_sigma_ratio,
    fft,
    hilbert,
    kurtosis,
    mean,
    median,
    phase,
    phase_band,
    q25,
    q75,
    rfft,
    rfft_bands,
    rms,
    skewness,
    spectrogram,
    std,
    zscore,
)
from .fft import fft_amps, fft_powers
from .filters import (
    BandPasserCPUTorch,
    BandPassFilterTorch,
    bandpassfilter_np,
    gaussian_filter1d,
)
from .mne import to_dig_montage
from .PARAMS import BANDS_LIM_HZ_DICT
from .referencing import common_average  # random_reference
from .referencing import subtract_from_random_column
from .sampling import down_sample_1d
from .transform import arr2skdf, crop, psd, to_z, wavelet, wavelet_np
