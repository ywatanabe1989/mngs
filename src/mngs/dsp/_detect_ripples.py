#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-02 23:31:11 (ywatanabe)"
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


def detect_ripples(
    xx,
    fs,
    low_hz=80,
    high_hz=140,
    sd=2.0,
    smoothing_sigma_ms=4,
    min_duration_ms=10,
):
    """
    xx: 2-dimensional (n_chs, seq_len) or 3-dimensional (batch_size, n_chs, seq_len)
    """

    if xx.ndim == 2:
        xx = xx[np.newaxis]

    RIPPLE_BANDS = np.vstack([[low_hz, high_hz]])

    # Bandpass Filtering
    xx = (
        (
            mngs.dsp.filt.bandpass(
                np.array(xx).astype(float),
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

    # Detects peaks of RMS values
    dfs = []
    for ii in range(len(xx)):
        xi = xx[ii]

        # Finds peaks over the designated standard deviation
        peaks, properties = find_peaks(xi, height=sd)

        # Determines the range around each peak (customize as needed)
        peak_ranges = []

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
                peak_ranges.append((left_ips, right_ips))

        # Converts to DataFrame
        starts, ends = zip(*peak_ranges) if peak_ranges else ([], [])
        df = (
            pd.DataFrame(
                data=np.vstack([starts, ends]).T, columns=["start_s", "end_s"]
            )
            / fs
        )
        df["duration_s"] = df.end_s - df.start_s

        # Filters events with short duration
        df = df[df.duration_s > (min_duration_ms * 1e-3)]

        # Adds indices
        df["index"] = ii

        # Sorts columns
        df = mngs.gen.mv_col(df, "index", 0)

        dfs.append(df)

    return pd.concat(dfs).set_index("index")


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
