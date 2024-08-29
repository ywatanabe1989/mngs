# [`mngs.dsp`](https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/dsp/)

## Overview
The `mngs.dsp` module provides Digital Signal Processing (DSP) utilities written in **PyTorch**, optimized for **CUDA** devices when available. This module offers efficient implementations of various DSP algorithms and techniques.

## Installation
```bash
pip install mngs
```

## Features
- PyTorch-based implementations for GPU acceleration
- Wavelet transforms and analysis
- Filtering operations (e.g., bandpass, lowpass, highpass)
- Spectral analysis tools
- Time-frequency analysis utilities
- Signal generation and manipulation functions
- Phase-Amplitude Coupling (PAC) analysis
- Modulation Index calculation
- Hilbert transform
- Power Spectral Density (PSD) estimation
- Resampling utilities

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
```python
import mngs
import torch
import numpy as np

# Generate a sample signal
t = torch.linspace(0, 1, 1000)
signal = torch.sin(2 * np.pi * 10 * t) + 0.5 * torch.sin(2 * np.pi * 20 * t)

# Perform wavelet transform
wavelet_coeffs = mngs.dsp.wavelet_transform(signal, wavelet='db4', level=5)

# Apply bandpass filter
filtered_signal = mngs.dsp.bandpass_filter(signal, lowcut=5, highcut=15, fs=1000)

# Compute spectrogram
freqs, times, Sxx = mngs.dsp.spectrogram(signal, fs=1000, nperseg=256)

# Generate chirp signal
chirp = mngs.dsp.chirp(t, f0=1, f1=20, method='linear')

# Perform Hilbert transform
analytic_signal = mngs.dsp.hilbert(signal)

# Calculate Phase-Amplitude Coupling
pac = mngs.dsp.phase_amplitude_coupling(signal, fs=1000)

# Estimate Power Spectral Density
freqs, psd = mngs.dsp.psd(signal, fs=1000)

# Resample signal
resampled_signal = mngs.dsp.resample(signal, orig_fs=1000, new_fs=500)
```

## API Reference
- `mngs.dsp.wavelet_transform(signal, wavelet, level)`: Performs wavelet transform
- `mngs.dsp.bandpass_filter(signal, lowcut, highcut, fs)`: Applies bandpass filter
- `mngs.dsp.lowpass_filter(signal, cutoff, fs)`: Applies lowpass filter
- `mngs.dsp.highpass_filter(signal, cutoff, fs)`: Applies highpass filter
- `mngs.dsp.spectrogram(signal, fs, nperseg)`: Computes spectrogram
- `mngs.dsp.stft(signal, fs, nperseg)`: Performs Short-Time Fourier Transform
- `mngs.dsp.istft(stft, fs, nperseg)`: Performs Inverse Short-Time Fourier Transform
- `mngs.dsp.chirp(t, f0, f1, method)`: Generates chirp signal
- `mngs.dsp.hilbert(signal)`: Performs Hilbert transform
- `mngs.dsp.phase_amplitude_coupling(signal, fs)`: Calculates Phase-Amplitude Coupling
- `mngs.dsp.modulation_index(signal, fs)`: Computes Modulation Index
- `mngs.dsp.psd(signal, fs)`: Estimates Power Spectral Density
- `mngs.dsp.resample(signal, orig_fs, new_fs)`: Resamples signal to new frequency
- `mngs.dsp.add_noise(signal, snr)`: Adds noise to signal with specified SNR

## Use Cases
- Audio signal processing
- Biomedical signal analysis
- Vibration analysis
- Communication systems
- Radar and sonar signal processing
- Neuroscience data analysis

## Performance
The `mngs.dsp` module leverages PyTorch's GPU acceleration capabilities, providing significant speedups for large-scale signal processing tasks when run on CUDA-enabled devices.

## Contributing
Contributions to improve `mngs.dsp` are welcome. Please submit pull requests or open issues on the GitHub repository.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
Yusuke Watanabe (ywata1989@gmail.com)

For more information and updates, please visit the [mngs GitHub repository](https://github.com/ywatanabe1989/mngs).

