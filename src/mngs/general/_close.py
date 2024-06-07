#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-06 08:13:23 (ywatanabe)"

import os
from datetime import datetime
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


def close(CONFIG, message=":)", notify=True, verbose=True):
    try:
        CONFIG["END_TIME"] = datetime.now()
        CONFIG["SPENT_TIME"] = format_diff_time(
            CONFIG["END_TIME"] - CONFIG["START_TIME"]
        )
        if verbose:
            print(f"\nEND TIME: {CONFIG['END_TIME']}")
            print(f"\nSPENT TIME: {CONFIG['SPENT_TIME']}")

    except Exception as e:
        print(e)

    mngs.io.save(
        CONFIG, CONFIG["SDIR"] + "CONFIGS/CONFIG.pkl", verbose=verbose
    )
    mngs.io.save(CONFIG, CONFIG["SDIR"] + "CONFIGS/CONFIG.yaml", verbose=False)

    try:
        if CONFIG.get("DEBUG", False):
            message = f"[DEBUG]\n" + message
        sleep(1)
        if notify:
            mngs.gen.notify(
                message=message,
                ID=CONFIG["ID"],
                log_paths=glob(CONFIG["SDIR"] + "logs/*.log"),
                verbose=verbose,
            )
    except Exception as e:
        print(e)

    # RUNNING to FINISHED
    finish(CONFIG["SDIR"])


def finish(src_dir, remove_src_dir=True):
    dest_dir = src_dir.replace("RUNNING/", "FINISHED/")
    os.makedirs(dest_dir, exist_ok=True)
    try:
        os.rename(src_dir, dest_dir)
        mngs.gen.print_block(
            f"Congratulations! The script completed.\n\n{dest_dir}",
            c="yellow",
        )
        if remove_src_dir:
            mngs.sh(f"rm {src_dir} -rf", verbose=False)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import mngs
    from icecream import ic

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )

    ic("aaa")
    ic("bbb")
    ic("ccc")

    close(CONFIG)
