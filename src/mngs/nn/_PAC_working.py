#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-08 11:40:19 (ywatanabe)"

from mngs.nn import BandPassFilter, Hilbert, ModulationIndex


def calc_bands(pha_or_amp, res_str="hres", start_hz=None, end_hz=None):
    def calc_bands_pha(res_str="hres", start_hz=2, end_hz=20):
        start_hz = start_hz if start_hz is not None else 2
        end_hz = end_hz if end_hz is not None else 20
        n_bands = dict(lres=10, mres=30, hres=50, demon=70, hulk=100)[res_str]
        mid_hz = np.linspace(start_hz, end_hz, n_bands)
        return np.c_[mid_hz - mid_hz / 4.0, mid_hz + mid_hz / 4.0]

    def calc_bands_amp(res_str="hres", start_hz=30, end_hz=160):
        start_hz = start_hz if start_hz is not None else 30
        end_hz = end_hz if end_hz is not None else 160
        n_bands = dict(lres=10, mres=30, hres=50, demon=70, hulk=100)[res_str]
        mid_hz = np.linspace(start_hz, end_hz, n_bands)
        return np.c_[mid_hz - mid_hz / 8.0, mid_hz + mid_hz / 8.0]

    if "pha" in pha_or_amp:
        return calc_bands_pha(
            res_str=res_str, start_hz=start_hz, end_hz=end_hz
        )

    if "amp" in pha_or_amp:
        return calc_bands_amp(
            res_str=res_str, start_hz=start_hz, end_hz=end_hz
        )


# Parameters
BANDS = mngs.dsp.PARAMS.BANDS
I_BATCH_SIZE = 0
I_CHS = 1
I_FREQS = 2
I_SEGMENTS = 3
I_SEQ_LEN = 4
BANDS_PHA = calc_bands("phase", res_str="hulk")
BANDS_AMP = calc_bands("amp", res_str="hulk")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FS = 512
T_SEC = 4


# low_hz, high_hz = mngs.dsp.PARAMS.BANDS["delta"]
xx, tt, fs = mngs.dsp.demo_sig(
    batch_size=1, n_chs=1, fs=FS, t_sec=T_SEC, sig_type="tensorpac"
)
xx = torch.tensor(xx).to(DEVICE)  # (batch_size, n_chs, n_segments, seq_len)
# xx = xx.unsqueeze(-2) # adds n_segments for chirp


# Preparing functions
bandpass_filters_pha = [
    BandPassFilter(ll, hh, fs, kernel_size=None).to(DEVICE)
    for ll, hh in BANDS_PHA
]
bandpass_filters_amp = [
    BandPassFilter(ll, hh, fs, kernel_size=None).to(DEVICE)
    for ll, hh in BANDS_AMP
]
hilbert = Hilbert(dim=-1).to(DEVICE)
modulation_index = ModulationIndex(n_bins=18).to(DEVICE)

# Calculation
batch_size, n_chs, n_segments, seq_len = xx.shape
# pha.shape: (batch_size, n_channels, n_freqs_pha, n_segments, sequence_length)
# amp.shape: (batch_size, n_channels, n_freqs_amp, n_segments, sequence_length)
pha = (
    torch.stack(
        [hilbert(bf_pha(xx.squeeze(0))) for bf_pha in bandpass_filters_pha],
        dim=I_FREQS,
    )[..., 0]
    .unsqueeze(I_CHS)
    .transpose(I_FREQS, I_SEGMENTS)
)

amp = (
    torch.stack(
        [hilbert(bf_amp(xx.squeeze(0))) for bf_amp in bandpass_filters_amp],
        dim=I_FREQS,
    )[..., 1]
    .unsqueeze(I_CHS)
    .transpose(I_FREQS, I_SEGMENTS)
)
pac = modulation_index(pha, amp)

# Plots PAC, the final output
i_batch = 0
i_ch = 0
fig, ax = mngs.plt.subplots()
ax.imshow2d(
    pac[i_batch, i_ch].cpu().numpy(),
)
ax = mngs.plt.ax.set_ticks(
    ax,
    xticks=np.array(BANDS_PHA).mean(axis=-1).astype(int),
    yticks=np.array(BANDS_AMP).mean(axis=-1).astype(int),
)
ax = mngs.plt.ax.set_n_ticks(ax)
ax.set_xlabel("Frequency for phase [Hz]")
ax.set_ylabel("Frequency for amplitude [Hz]")
ax.set_title("PAC values")
plt.show()
