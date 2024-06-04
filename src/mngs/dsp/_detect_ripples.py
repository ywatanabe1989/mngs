#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-03 11:23:10 (ywatanabe)"
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


def _detect_ripples_preprocess(xx, fs, low_hz, high_hz, smoothing_sigma_ms=4):
    if xx.ndim == 2:
        xx = xx[np.newaxis]

    # Downsampling
    fs_tgt = low_hz * 3
    xx = mngs.dsp.resample(xx, float(fs), float(fs_tgt))
    fs = fs_tgt  # override

    # Subtracts the global mean to reduce false detection due to EMG signal
    xx -= np.nanmean(xx, axis=1, keepdims=True)

    RIPPLE_BANDS = np.vstack([[low_hz, high_hz]])

    # Bandpass Filtering
    xx = (
        (
            mngs.dsp.filt.bandpass(
                np.array(xx),
                fs,
                RIPPLE_BANDS,
            )
        )
        .squeeze(-2)
        .astype(np.float64)
    )

    # Calculate RMS
    xx = xx**2
    _, xx = mngs.dsp.hilbert(xx)
    xx = mngs.dsp.filt.gauss(xx, smoothing_sigma_ms * 1e-3 * fs).squeeze(-2)
    xx = np.sqrt(xx)

    # Scales across channels
    xx = xx.mean(axis=1)
    xx = mngs.gen.to_z(xx, dim=-1)
    return xx, fs_tgt


def detect_ripples(
    xx,
    fs,
    low_hz,
    high_hz,
    sd=2.0,
    smoothing_sigma_ms=4,
    min_duration_ms=10,
):
    """
    xx: 2-dimensional (n_chs, seq_len) or 3-dimensional (batch_size, n_chs, seq_len)
    """

    try:
        xx, _ = _detect_ripples_preprocess(
            xx, fs, low_hz, high_hz, smoothing_sigma_ms
        )

        # Detects peaks of RMS values
        dfs = []
        for ii in range(len(xx)):
            xi = xx[ii]

            # Finds peaks over the designated standard deviation
            peaks, properties = find_peaks(xi, height=sd)

            # Determines the range around each peak (customize as needed)
            peaks_all = []
            peak_ranges = []
            peak_amplitudes_sd = []

            for peak in peaks:
                left_bound = np.where(xi[:peak] < 0)[0]
                right_bound = np.where(xi[peak:] < 0)[0]

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
                    peak_amplitudes_sd.append(xi[peak])

            # Converts to DataFrame
            if peak_ranges:
                starts, ends = zip(*peak_ranges) if peak_ranges else ([], [])
                df = pd.DataFrame(
                    {
                        "start_s": np.hstack(starts) / fs,
                        "peak_s": np.hstack(peaks_all) / fs,
                        "end_s": np.hstack(ends) / fs,
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

            # Relative peak
            delta_s = df.peak_s - df.start_s
            rel_peak = delta_s / df.duration_s
            df["relative_peak_pos"] = np.round(rel_peak, 3)

            # Index
            df.index = [ii for _ in range(len(df))]
            # # Adds indices
            # df["index"] = ii
            # # Sorts columns
            # df = mngs.gen.mv_col(df, "index", 0)

            # Exclude edge effects
            edge_s = 1 / low_hz * 3
            indi_drop = (df.start_s < edge_s) + (
                xx.shape[-1] / fs - edge_s < df.end_s
            )
            df = df[~indi_drop]

            # Incidence [Hz]
            n_ripples = len(df)
            rec_s = xx.shape[-1] / fs
            df["incidence_hz"] = n_ripples / rec_s

            # Sorting
            sorted_columns = [
                "start_s",
                "end_s",
                "duration_s",
                "peak_s",
                "relative_peak_pos",
                "peak_amp_sd",
                "incidence_hz",
            ]
            df = df[sorted_columns]

            # Summarize as a dataframe across batch
            dfs.append(df)

        return pd.concat(dfs)  # .set_index("index")

    except ValueError as e:
        print("Caught an error:", e)
    # except Exception as e:
    #     print("Something wrong with ripple detection. :", e)


def main():
    xx, tt, fs = mngs.dsp.demo_sig(sig_type="ripple")
    df = detect_ripples(xx, fs)
    print(df)


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
