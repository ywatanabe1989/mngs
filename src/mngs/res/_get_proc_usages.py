#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-22 23:26:53"
# Author: Yusuke Watanabe (ywata1989@gmail.com)

"""
This script does XYZ.
"""

# Functions
import os
import subprocess
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import mngs
import pandas as pd
import psutil


# Functions
def get_proc_usages():
    """
    Retrieves the current usage statistics for the CPU, RAM, GPU, and VRAM.

    This function fetches the current usage percentages for the CPU and GPU, as well as the current usage in GiB for RAM and VRAM.
    The data is then compiled into a pandas DataFrame with the current timestamp.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the current usage statistics with the following columns:
                      - Time: The timestamp when the data was retrieved.
                      - CPU [%]: The CPU usage percentage.
                      - RAM [GiB]: The RAM usage in GiB.
                      - GPU [%]: The GPU usage percentage.
                      - VRAM [GiB]: The VRAM usage in GiB.
                      Each row in the DataFrame represents a single instance of data retrieval, rounded to 1 decimal place.

    Example:
        >>> usage_df = get_proc_usages()
        >>> print(usage_df)
    """
    cpu_perc, ram_gb = _get_cpu_usage()
    gpu_perc, vram_gb = _get_gpu_usage()

    sr = pd.Series(
        {
            "Time": datetime.now(),
            "CPU [%]": cpu_perc,
            "RAM [GiB]": ram_gb,
            "GPU [%]": gpu_perc,
            "VRAM [GiB]": vram_gb,
        }
    )

    df = pd.DataFrame(sr).round(1).T

    return df


def _get_cpu_usage(process=os.getpid(), n_round=1):
    cpu_usage_perc = psutil.cpu_percent()
    ram_usage_gb = (
        psutil.virtual_memory().percent
        / 100
        * psutil.virtual_memory().total
        / (1024**3)
    )
    return round(cpu_usage_perc, n_round), round(ram_usage_gb, n_round)


def _get_gpu_usage(n_round=1):
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,nounits,noheader",
        ],
        capture_output=True,
        text=True,
    )
    gpu_usage_perc, _vram_usage_mib = result.stdout.strip().split(",")
    vram_usage_gb = float(_vram_usage_mib) / 1024
    return round(float(gpu_usage_perc), n_round), round(
        float(vram_usage_gb), n_round
    )


# (YOUR AWESOME CODE)

if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )

    usage = mngs.res.get_proc_usages()
    mngs.io.save(usage, "usage.csv")

    # Close
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF

"""
/home/ywatanabe/proj/entrance/mngs/res/_get_procs_usage.py
"""
