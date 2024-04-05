#!/usr/bin/env python3

from . import PARAMS, add_noise, filt, norm, ref
from ._demo_sig import demo_sig

# from ._ensure_3d import ensure_3d
from ._hilbert import hilbert
from ._misc import ensure_3d
from ._mne import get_eeg_pos
from ._psd import psd
from ._resample import resample
from ._transform import to_segments, to_sktime_df
from ._wavelet import wavelet
