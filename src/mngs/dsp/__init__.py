#!/usr/bin/env python3

# from .wavelet import wavelet
from . import filt, np, thinkdsp
from ._ensure_3d import ensure_3d

# from ._demo_sig import demo_sig_np, demo_sig_torch
from ._psd import psd_torch
from ._wavelet import wavelet
from .feature_extractors import (  # kurtosis,; mean,; median,; q25,; q75,; skewness,; std,; zscore,; bandpass,; bandstop,; hilbert,
    FeatureExtractorTorch,
    amp,
    amp_band,
    beyond_r_sigma_ratio,
    fft,
    phase,
    phase_band,
    rfft,
    rfft_bands,
    rms,
    spectrogram,
)

# from .filters import (
#     BandPasserCPUTorch,
#     BandPassFilterTorch,
#     bandpassfilter_np,
#     gaussian_filter1d,
# )
from .mne import to_dig_montage
from .PARAMS import BANDS_LIM_HZ_DICT
from .referencing import common_average  # random_reference
from .referencing import subtract_from_random_column
from .sampling import down_sample_1d
from .transform import arr2skdf, crop, psd, to_z, wavelet, wavelet_np

# from ._filter import (
#     BandPasserCPUTorch,
#     BandPassFilterTorch,
#     bandpass,
#     bandpassfilter_np,
#     bandstop,
#     gaussian_filter1d,
# )


# from .fft import fft_amps, fft_powers
