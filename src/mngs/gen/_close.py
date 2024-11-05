#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 23:19:10 (ywatanabe)"
# File: ./mngs_repo/src/mngs/gen/_close.py

import os
import re
import shutil
import time
from datetime import datetime
from glob import glob as glob

from ..io import save as mngs_io_save
from ..io import flush as mngs_io_flush
from ..utils._notify import notify as mngs_gen_notify
from ..str._printc import printc


def _format_diff_time(diff_time):
    # Get total seconds from the timedelta object
    total_seconds = int(diff_time.total_seconds())

    # Calculate hours, minutes and seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    # Format the time difference as a string
    diff_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return diff_time_str


def _process_timestamp(CONFIG, verbose=True):
    try:
        CONFIG["END_TIME"] = datetime.now()
        CONFIG["RUN_TIME"] = _format_diff_time(
            CONFIG["END_TIME"] - CONFIG["START_TIME"]
        )
        if verbose:
            print()
            print(f"START TIME: {CONFIG['START_TIME']}")
            print(f"END TIME: {CONFIG['END_TIME']}")
            print(f"RUN TIME: {CONFIG['RUN_TIME']}")
            print()

    except Exception as e:
        print(e)

    return CONFIG


def _save_configs(CONFIG):
    mngs_io_save(CONFIG, CONFIG["SDIR"] + "CONFIGS/CONFIG.pkl", verbose=False)
    mngs_io_save(CONFIG, CONFIG["SDIR"] + "CONFIGS/CONFIG.yaml", verbose=False)


def _escape_ANSI_from_log_files(log_files):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    # ANSI code escape
    for f in log_files:
        with open(f, "r", encoding="utf-8") as file:
            content = file.read()
        cleaned_content = ansi_escape.sub("", content)
        with open(f, "w", encoding="utf-8") as file:
            file.write(cleaned_content)


def close(
    CONFIG, message=":)", notify=True, verbose=True, exit_status=None
):
    try:
        CONFIG.EXIT_STATUS = exit_status
        CONFIG = CONFIG.to_dict()
        CONFIG = _process_timestamp(CONFIG, verbose=verbose)
        sys = CONFIG.pop("sys")
        _save_configs(CONFIG)
        mngs_io_flush(sys=sys)

        # RUNNING to RUNNING2FINISHEDED
        CONFIG = running2finished(CONFIG, exit_status=exit_status)
        mngs_io_flush(sys=sys)

        # ANSI code escape
        log_files = _glob(CONFIG["SDIR"] + "logs/*.log")
        _escape_ANSI_from_log_files(log_files)
        mngs_io_flush(sys=sys)

        if notify:
            try:
                message = (
                    f"[DEBUG]\n" + str(message)
                    if CONFIG.get("DEBUG", False)
                    else str(message)
                )
                mngs_gen_notify(
                    message=message,
                    ID=CONFIG["ID"],
                    attachment_paths=log_files,
                    verbose=verbose,
                )
                mngs_io_flush(sys=sys)
            except Exception as e:
                print(e)

    finally:
        # Ensure file handles are closed
        if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'close'):
            sys.stdout.close()
        if hasattr(sys, 'stderr') and hasattr(sys.stderr, 'close'):
            sys.stderr.close()
    # try:
    #     sys.stdout.close()
    #     sys.stderr.close()
    # except Exception as e:
    #     print(e)


def running2finished(
    CONFIG, exit_status=None, remove_src_dir=True, max_wait=60
):
    if exit_status == 0:
        dest_dir = CONFIG["SDIR"].replace("RUNNING/", "FINISHED_SUCCESS/")
    elif exit_status == 1:
        dest_dir = CONFIG["SDIR"].replace("RUNNING/", "FINISHED_ERROR/")
    else:  # exit_status is None:
        dest_dir = CONFIG["SDIR"].replace("RUNNING/", "FINISHED/")

    src_dir = CONFIG["SDIR"]
    # if dest_dir is None:
    #     dest_dir = src_dir.replace("RUNNING/", "FINISHED/")

    os.makedirs(dest_dir, exist_ok=True)
    try:

        # Copy files individually
        for item in os.listdir(src_dir):
            s = os.path.join(src_dir, item)
            d = os.path.join(dest_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

        start_time = time.time()
        while (
            not os.path.exists(dest_dir)
            and time.time() - start_time < max_wait
        ):
            time.sleep(0.1)
        if os.path.exists(dest_dir):
            printc(
                f"Congratulations! The script completed.\n\n{dest_dir}",
                c="yellow",
            )
            if remove_src_dir:
                shutil.rmtree(src_dir)
        else:
            print(f"Copy operation timed out after {max_wait} seconds")

        CONFIG["SDIR"] = dest_dir
    except Exception as e:
        print(e)

    finally:
        return CONFIG


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt

    from icecream import ic

    from .._start import start


    CONFIG, sys.stdout, sys.stderr, plt, CC = start(
        sys, plt, verbose=False
    )

    ic("aaa")
    ic("bbb")
    ic("ccc")

    close(CONFIG)


# EOF
