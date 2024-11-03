#!/usr/bin/env python3

from . import PARAMS, add_noise, filt, norm, reference, utils
from ._crop import crop
from ._demo_sig import demo_sig
from ._detect_ripples import detect_ripples
from ._hilbert import hilbert
from ._misc import ensure_3d
from ._mne import get_eeg_pos
from ._modulation_index import modulation_index
from ._pac import pac
from ._psd import psd
from ._resample import resample
from ._time import time
from ._transform import to_segments, to_sktime_df
from ._wavelet import wavelet

# try:
#     from . import PARAMS, add_noise, filt, norm, reference, utils
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import some modules. Error: {e}")

# try:
#     from ._crop import crop
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import crop. Error: {e}")

# try:
#     from ._demo_sig import demo_sig
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import demo_sig. Error: {e}")

# try:
#     from ._detect_ripples import detect_ripples
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import detect_ripples. Error: {e}")

# try:
#     from ._hilbert import hilbert
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import hilbert. Error: {e}")

# try:
#     from ._misc import ensure_3d
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import ensure_3d. Error: {e}")

# try:
#     from ._mne import get_eeg_pos
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import get_eeg_pos. Error: {e}")

# try:
#     from ._modulation_index import modulation_index
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import modulation_index. Error: {e}")

# try:
#     from ._pac import pac
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import pac. Error: {e}")

# try:
#     from ._psd import psd
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import psd. Error: {e}")

# try:
#     from ._resample import resample
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import resample. Error: {e}")

# try:
#     from ._time import time
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import time. Error: {e}")

# try:
#     from ._transform import to_segments, to_sktime_df
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import to_segments or to_sktime_df. Error: {e}")

# try:
#     from ._wavelet import wavelet
# except ImportError as e:
#     warnings.warn(f"Warning: Failed to import wavelet. Error: {e}")

# # #!/usr/bin/env python3


# # from . import PARAMS, add_noise, filt, norm, reference, utils
# # from ._crop import crop
# # from ._demo_sig import demo_sig
# # from ._detect_ripples import detect_ripples

# # # from ._ensure_3d import ensure_3d
# # from ._hilbert import hilbert

# # # from ._listen import listen
# # from ._misc import ensure_3d
# # from ._mne import get_eeg_pos
# # from ._modulation_index import modulation_index
# # from ._pac import pac
# # from ._psd import psd
# # from ._resample import resample
# # from ._time import time
# # from ._transform import to_segments, to_sktime_df
# # from ._wavelet import wavelet
