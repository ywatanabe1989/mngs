#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 11:23:25 (ywatanabe)"
# File: ./mngs_repo/src/mngs/resource/_log_processor_usages.py

"""
Functionality:
    * Monitors and logs system resource utilization over time
Input:
    * Path for saving logs
    * Monitoring duration and interval
Output:
    * CSV file containing time-series resource usage data
Prerequisites:
    * mngs package with processor usage monitoring capabilities
"""

"""Imports"""
import math
import os
import sys
import time
from multiprocessing import Process
from typing import Union

import matplotlib.pyplot as plt
import mngs
import pandas as pd

from .._sh import sh
from ..io._load import load
from ..io._save import save
from ..str import printc
from ._get_processor_usages import get_processor_usages

"""Functions & Classes"""
def log_processor_usages(
    path: str = "/tmp/mngs/processor_usages.csv",
    limit_min: float = 30,
    interval_s: float = 1,
    init: bool = True,
    verbose: bool = False,
    background: bool = False,
) -> Union[None, Process]:
    """Logs system resource usage over time.

    Parameters
    ----------
    path : str
        Path to save the log file
    limit_min : float
        Monitoring duration in minutes
    interval_s : float
        Sampling interval in seconds
    init : bool
        Whether to clear existing log file
    verbose : bool
        Whether to print the log
    background : bool
        Whether to run in background

    Returns
    -------
    Union[None, Process]
        Process object if background=True, None otherwise
    """
    if background:
        process = Process(
            target=_log_processor_usages,
            args=(path, limit_min, interval_s, init, verbose)
        )
        process.start()
        return process

    return _log_processor_usages(
        path=path,
        limit_min=limit_min,
        interval_s=interval_s,
        init=init,
        verbose=verbose,
    )

def _log_processor_usages(
    path: str = "/tmp/mngs/processor_usages.csv",
    limit_min: float = 30,
    interval_s: float = 1,
    init: bool = True,
    verbose: bool = False,
) -> None:
    """Logs system resource usage over time.

    Parameters
    ----------
    path : str
        Path to save the log file
    limit_min : float
        Monitoring duration in minutes
    interval_s : float
        Sampling interval in seconds
    init : bool
        Whether to clear existing log file
    verbose : bool
        Whether to print the log

    Example
    -------
    >>> log_processor_usages(path="usage_log.csv", limit_min=5)
    """
    assert path.endswith(".csv"), "Path must end with .csv"

    # Log file initialization
    _ensure_log_file(path, init)
    printc(f"Log file can be monitored with with `tail -f {path}`")

    limit_s = limit_min * 60
    n_max = math.ceil(limit_s // interval_s)

    for _ in range(n_max):
        _add(path, verbose=verbose)
        time.sleep(interval_s)

def _ensure_log_file(path: str, init: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        printc(f"{path} was newly created.")
        save(pd.DataFrame(), path, verbose=False)

    if init and os.path.exists(path):
        try:
            sh(f"rm {path}")
            save(pd.DataFrame(), path, verbose=False)
            printc(f"{path} was newly created.")
        except Exception as err:
            raise RuntimeError(f"Failed to init log file: {err}")

def _add(path: str, verbose: bool = True) -> None:
    try:
        past = load(path) if os.path.exists(path) else pd.DataFrame()
        now = get_processor_usages()

        # Reset index before concatenation to ensure continuity
        if not past.empty:
            now.index = [past.index.max() + 1]

        combined = pd.concat([past, now]).round(3)
        save(combined, path, verbose=verbose)

        if verbose:
            print(f"\n{combined}")
    except Exception as err:
        raise RuntimeError(f"Failed to update log: {err}")

main = log_processor_usages

if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt, verbose=False)
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# python -c "import mngs; mngs.resource.log_processor_usages(\"/tmp/processor_usages.csv\", init=True)"

# EOF
