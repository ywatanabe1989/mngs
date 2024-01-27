#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-27 19:24:23 (ywatanabe)"

from datetime import datetime, timedelta
from glob import glob
from time import sleep

import mngs


def format_diff_time(diff_time):
    # Get total seconds from the timedelta object
    total_seconds = int(diff_time.total_seconds())

    # Calculate hours, minutes and seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    # Format the time difference as a string
    diff_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return diff_time_str


def close(CONFIG, message=":)", show=True):
    try:
        end_time = datetime.now()
        diff_time = format_diff_time(end_time - CONFIG["START_TIME"])
        CONFIG["TimeSpent"] = diff_time
        del CONFIG["START_TIME"]
    except Exception as e:
        print(e)

    mngs.io.save(CONFIG, CONFIG["SDIR"] + "CONFIG.pkl")

    try:
        sleep(3)
        mngs.gen.notify(
            message=f"[DEBUG]\n" + message,
            ID=CONFIG["ID"],
            log_paths=glob(CONFIG["SDIR"] + "*.log"),
            show=show,
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt = mngs.gen.start(sys, plt, show=False)

    ic("aaa")
    ic("bbb")
    ic("ccc")

    close(CONFIG)
