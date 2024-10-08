#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-05 08:25:07 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/dsp/_detect_ripples.py


"""
This script does XYZ.
"""


"""
Imports
"""
import sys

import matplotlib.pyplot as plt
import mngs

mngs.gen.reload(mngs)

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def main():
    xx, tt, fs = mngs.dsp.demo_sig(sig_type="ripple")
    df = detect_ripples(xx, fs, 80, 140)
    print(df)


def detect_ripples(
    xx,
    fs,
    low_hz,
    high_hz,
    sd=2.0,
    smoothing_sigma_ms=4,
    min_duration_ms=10,
    return_preprocessed_signal=False,
):
    """
    xx: 2-dimensional (n_chs, seq_len) or 3-dimensional (batch_size, n_chs, seq_len) wide-band signal.
    """

    try:
        xx_r, fs_r = _preprocess(xx, fs, low_hz, high_hz, smoothing_sigma_ms)
        df = _find_events(xx_r, fs_r, sd, min_duration_ms)
        df = _drop_ripples_at_edges(df, low_hz, xx_r, fs_r)
        df = _calc_relative_peak_position(df)
        # df = _calc_incidence(df, xx_r, fs_r)
        df = _sort_columns(df)

        if not return_preprocessed_signal:
            return df

        elif return_preprocessed_signal:
            return df, xx_r, fs_r

    except ValueError as e:
        print("Caught an error:", e)


def _preprocess(xx, fs, low_hz, high_hz, smoothing_sigma_ms=4):
    # Ensures three dimensional
    if xx.ndim == 2:
        xx = xx[np.newaxis]
    assert xx.ndim == 3

    # For readability
    RIPPLE_BANDS = np.vstack([[low_hz, high_hz]])

    # Downsampling
    fs_tgt = low_hz * 3
    xx = mngs.dsp.resample(xx, float(fs), float(fs_tgt))
    fs = fs_tgt

    # Subtracts the global mean to reduce false detection due to EMG signal
    xx -= np.nanmean(xx, axis=1, keepdims=True)

    # Bandpass Filtering
    xx = (
        (
            mngs.dsp.filt.bandpass(
                np.array(xx),
                fs_tgt,
                RIPPLE_BANDS,
            )
        )
        .squeeze(-2)
        .astype(np.float64)
    )

    # Calculate RMS
    xx = xx**2
    _, xx = mngs.dsp.hilbert(xx)
    xx = mngs.dsp.filt.gauss(xx, smoothing_sigma_ms * 1e-3 * fs_tgt).squeeze(
        -2
    )
    xx = np.sqrt(xx)

    # Scales across channels
    xx = xx.mean(axis=1)
    xx = mngs.gen.to_z(xx, dim=-1)

    return xx, fs_tgt


def _find_events(xx_r, fs_r, sd, min_duration_ms):
    def _find_events_1d(xx_ri, fs_r, sd, min_duration_ms):
        # Finds peaks over the designated standard deviation
        peaks, properties = find_peaks(xx_ri, height=sd)

        # Determines the range around each peak (customize as needed)
        peaks_all = []
        peak_ranges = []
        peak_amplitudes_sd = []

        for peak in peaks:
            left_bound = np.where(xx_ri[:peak] < 0)[0]
            right_bound = np.where(xx_ri[peak:] < 0)[0]

            left_ips = left_bound.max() if left_bound.size > 0 else peak
            right_ips = (
                peak + right_bound.min() if right_bound.size > 0 else peak
            )

            # Avoid duplicates: Check if the current peak range is already listed
            if not any(
                (left_ips == start and right_ips == end)
                for start, end in peak_ranges
            ):
                peaks_all.append(peak)
                peak_ranges.append((left_ips, right_ips))
                peak_amplitudes_sd.append(xx_ri[peak])

        # Converts to DataFrame
        if peak_ranges:
            starts, ends = zip(*peak_ranges) if peak_ranges else ([], [])
            df = pd.DataFrame(
                {
                    "start_s": np.hstack(starts) / fs_r,
                    "peak_s": np.hstack(peaks_all) / fs_r,
                    "end_s": np.hstack(ends) / fs_r,
                    "peak_amp_sd": np.hstack(peak_amplitudes_sd),
                }
            ).round(3)
        else:
            df = pd.DataFrame(
                columns=["start_s", "peak_s", "end_s", "peak_amp_sd"]
            )

        # Duration
        df["duration_s"] = df.end_s - df.start_s

        # Filters events with short duration
        df = df[df.duration_s > (min_duration_ms * 1e-3)]

        return df

    if xx_r.ndim == 1:
        xx_r = xx_r[np.newaxis, :]
    assert xx_r.ndim == 2

    dfs = []
    for i_ch in range(len(xx_r)):
        xx_ri = xx_r[i_ch]
        df_i = _find_events_1d(xx_ri, fs_r, sd, min_duration_ms)
        df_i.index = [i_ch for _ in range(len(df_i))]
        dfs.append(df_i)
    dfs = pd.concat(dfs)

    return dfs


def _drop_ripples_at_edges(df, low_hz, xx_r, fs_r):
    edge_s = 1 / low_hz * 3
    indi_drop = (df.start_s < edge_s) + (
        xx_r.shape[-1] / fs_r - edge_s < df.end_s
    )
    df = df[~indi_drop]
    return df


def _calc_relative_peak_position(df):
    delta_s = df.peak_s - df.start_s
    rel_peak = delta_s / df.duration_s
    df["rel_peak_pos"] = np.round(rel_peak, 3)
    return df


# def _calc_incidence(df, xx_r, fs_r):
#     n_ripples = len(df)
#     rec_s = xx_r.shape[-1] / fs_r
#     df["incidence_hz"] = n_ripples / rec_s
#     return df


def _sort_columns(df):
    sorted_columns = [
        "start_s",
        "end_s",
        "duration_s",
        "peak_s",
        "rel_peak_pos",
        "peak_amp_sd",
        # "incidence_hz",
    ]
    df = df[sorted_columns]
    return df


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF


# # append iEEG traces
# iEEG_traces = []
# for (
#     i_rip,
#     rip,
# ) in (
#     rip_df.reset_index().iterrows()
# ):  # rip_df = rips_df.iloc[i_trial]

#     start_pts = int(rip["start_time"] * CONFIG["FS_iEEG"])
#     end_pts = int(rip["end_time"] * CONFIG["FS_iEEG"])
#     # start_pts = int(rip["start_time"] * FS_TGT)
#     # end_pts = int(rip["end_time"] * FS_TGT)
#     iEEG_traces.append(
#         # iEEG[i_trial][:, start_pts:end_pts]
#         np.array(iEEG)[i_trial][:, start_pts:end_pts]
#     )
# rip_df["iEEG trace"] = iEEG_traces


# # append ripple band filtered iEEG traces
# ripple_band_iEEG_traces = []
# for (
#     i_rip,
#     rip,
# ) in (
#     rip_df.reset_index().iterrows()
# ):  # rip_df = rips_df.iloc[i_trial]
#     start_pts = int(rip["start_time"] * CONFIG["FS_iEEG"])
#     end_pts = int(rip["end_time"] * CONFIG["FS_iEEG"])

#     ripple_band_iEEG_traces.append(
#         iEEG_ripple_band_passed[i_trial][:, start_pts:end_pts]
#     )
# rip_df["ripple band iEEG trace"] = ripple_band_iEEG_traces

# # ripple peak amplitude
# ripple_peak_amplitude = [
#     np.abs(rbt).max(axis=-1) for rbt in ripple_band_iEEG_traces
# ]
# ripple_band_baseline_sd = iEEG_ripple_band_passed[i_trial].std(
#     axis=-1
# )
# rip_df["ripple_peak_amplitude_sd"] = [
#     (rpa / ripple_band_baseline_sd).mean()
#     for rpa in ripple_peak_amplitude
# ]

# rip_df["ripple_amplitude_sd"] = [
#     (np.abs(rbt).mean(axis=-1) / ripple_band_baseline_sd).mean()
#     for rbt in rip_df["ripple band iEEG trace"]
# ]
