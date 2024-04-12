# mngs.dsp
This mngs.dsp module is a digital signal processing toolbox developed in **PyTorch**. It performs calculations using torch.tensor. However, it also accommodates inputs in the form of **numpy.ndarray** and **pd.DataFrame** through wrapper functions. For example, if the input is a numpy.ndarray, it is initially converted to a torch.tensor (**utilizing CUDA automatically** when available), processed, and subsequently **returned in its original datatype** (numpy.ndarray). Additionally, [`mngs.dsp.nn`]("./dsp.nn") features PyTorch modules designed for straightforward integration into machine learning projects.

## Installation
```bash
$ pip install mngs
```

## Quick Start
``` python
import mngs
```

## Galleries
<div align="center">
  <img src="./_demo_sig/traces.png" height="400">
</div>

<div align="center">
  <img src="./_resampled/traces.png" height="400">
</div>

<div align="center">
  <img src="./filt/traces.png" height="400">
</div>

<div align="center">
  <img src="./filt/psd.png" height="400">
</div>

<div align="center">
  <img src="./_wavelet/wavelet.png" height="400">
</div>

<div align="center">
  <img src="./_hilbert/traces.png" height="400">
</div>

<div align="center">
  <img src="./_modulation_index/modulation_index.png" height="400">
</div>

<div align="center">
  <img src="./_pac/pac.png" height="400">
</div>





```python
# Parameters
SRC_FS = 1024  # Source sampling frequency
TGT_FS = 512   # Target sampling frequency
FREQS_HZ = [10, 30, 100]  # Frequencies in Hz
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
# xx is either of torch.tensor (on cpu / cuda), numpy.ndarray, or pd.DataFrame.

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
pac, freqs_pha, freqs_amp = mngs.dsp.pac(x_3d, fs) # This function is computationally demanding. Please monitor the RAM/VRAM usage.
```

## Contact
Yusuke Watanabe (ywata1989@gmail.com).
