# [`mngs.dsp`](https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/dsp/)
Digital signal processing utilities written in **PyTorch**, optimized for **CUDA** devices when available, with special attention to parallel computing.

It features:
1. Acceptance of **np.array** and **pd.DataFrame** as input, with internal calculations performed in PyTorch to leverage parallel and GPU computation, and results returned in numpy format for convenience, speed, and consistency.
2. The `torch.nn.modules` in [`mngs.dsp.nn`](https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/nn/) are ready for use in machine learning projects.

## Installation
```bash
$ pip install mngs
```

## Galleries
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
  <img src="./_demo_sig/traces.png" height="300" style="border: 2px solid gray; margin: 5px;">
  <img src="./_resample/traces.png" height="300" style="border: 2px solid gray; margin: 5px;">
  <img src="./filt/traces.png" height="300" style="border: 2px solid gray; margin: 5px;">
  <img src="./filt/psd.png" height="300" style="border: 2px solid gray; margin: 5px;">
</div>

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
  <img src="./_wavelet/wavelet.png" height="300" style="border: 2px solid gray; margin: 5px;">
  <img src="./_hilbert/traces.png" height="300" style="border: 2px solid gray; margin: 5px;">
  <img src="./_modulation_index/modulation_index.png" height="300" style="border: 2px solid gray; margin: 5px;">
  <img src="./_pac/pac_with_trainable_bandpass_fp32.png" height="300" style="border: 2px solid gray; margin: 5px;">
</div>


## Quick Start
``` python
import mngs

# Parameters
SRC_FS = 1024  # Source sampling frequency
TGT_FS = 512   # Target sampling frequency
FREQS_HZ = [10, 30, 100]  # Frequencies in Hz for periodic signals
LOW_HZ = 20    # Low frequency for bandpass filter
HIGH_HZ = 50   # High frequency for bandpass filter
SIGMA = 10     # Sigma for Gaussian filter
SIG_TYPES = [
    "uniform",
    "gauss",
    "periodic",
    "chirp",
    "ripple",
    "meg",
    "tensorpac",
] # Available signal types


# Demo Signal
xx, tt, fs = mngs.dsp.demo_sig(
    t_sec=T_SEC, fs=SRC_FS, freqs_hz=FREQS_HZ, sig_type="chirp"
)
# xx.shape (batch_size, n_chs, seq_len) 
# xx.shape (batch_size, n_chs, n_segments, seq_len) # when sig_type is "tensorpac" or "pac"


# # Various data types are automatically handled:
# xx = torch.tensor(xx).float()
# xx = torch.tensor(xx).float().cuda()
# xx = np.array(xx)
# xx = pd.DataFrame(xx)

# Normalization
xx_norm = mngs.dsp.norm.z(xx)
xx_minmax = mngs.dsp.norm.minmax(xx)

# Resampling
xx_resampled = mngs.dsp.resample(xx, fs, TGT_FS)

# Noise addition
xx_gauss = mngs.dsp.add_noise.gauss(xx)
xx_white = mngs.dsp.add_noise.white(xx)
xx_pink = mngs.dsp.add_noise.pink(xx)
xx_brown = mngs.dsp.add_noise.brown(xx)

# Filtering
xx_filted_bandpass = mngs.dsp.filt.bandpass(xx, fs, low_hz=LOW_HZ, high_hz=HIGH_HZ)
xx_filted_bandstop = mngs.dsp.filt.bandstop(xx, fs, low_hz=LOW_HZ, high_hz=HIGH_HZ)
xx_filted_gauss = mngs.dsp.filt.gauss(xx, sigma=SIGMA)

# Hilbert Transformation
phase, amplitude = mngs.dsp.hilbert(xx) # or envelope

# Wavelet Transformation
wavelet_coef, wavelet_freqs = mngs.dsp.wavelet(xx, fs)

# Power Spetrum Density
psd, psd_freqs = mngs.dsp.psd(xx, fs)

# Phase-Amplitude Coupling
pac, freqs_pha, freqs_amp = mngs.dsp.pac(xx, fs) # This process is computationally intensive. Please monitor RAM/VRAM usage.
```

## Contact
Yusuke Watanabe (ywata1989@gmail.com).
