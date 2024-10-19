#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Time-stamp: "2024-10-19 03:03:56 (ywatanabe)"
=======
# Time-stamp: "2024-10-19 04:40:35 (ywatanabe)"
>>>>>>> ff1c0838bf7bcd958d4181b20c2a42711bd4b920

import os
from datetime import datetime
from glob import glob
from time import sleep
import mngs
import time
import shutil
import re

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

<<<<<<< HEAD

def close(CONFIG, message=":)", notify=True, verbose=True, sys=None):
    CONFIG = CONFIG.to_dict()

=======
def process_timestamp(CONFIG, verbose=True):
>>>>>>> ff1c0838bf7bcd958d4181b20c2a42711bd4b920
    try:
        CONFIG["END_TIME"] = datetime.now()
        CONFIG["RUN_TIME"] = format_diff_time(
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

def save_configs(CONFIG):
    mngs.io.save(
        CONFIG, CONFIG["SDIR"] + "CONFIGS/CONFIG.pkl", verbose=False
    )
    mngs.io.save(CONFIG, CONFIG["SDIR"] + "CONFIGS/CONFIG.yaml", verbose=False)

def escape_ANSI_from_log_files(log_files):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    # ANSI code escape
    for f in log_files:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        cleaned_content = ansi_escape.sub('', content)
        with open(f, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)

def close(CONFIG, message=":)", notify=True, verbose=True, sys=None):
    CONFIG = CONFIG.to_dict()

    CONFIG = process_timestamp(CONFIG, verbose=verbose)

    save_configs(CONFIG)
    mngs.io.flush(sys=sys)

    # RUNNING to RUNNING2FINISHEDED
    running2finished(CONFIG["SDIR"])
    CONFIG["SDIR"] = CONFIG["SDIR"].replace("RUNNING", "FINISHED")
    mngs.io.flush(sys=sys)

    # ANSI code escape
    log_files = glob(CONFIG["SDIR"] + "logs/*.log")
    escape_ANSI_from_log_files(log_files)

    if notify:
        try:
            message = f"[DEBUG]\n" + str(message) if CONFIG.get("DEBUG", False) else str(message)
            mngs.gen.notify(
                message=message,
                ID=CONFIG["ID"],
                attachment_paths=log_files,
                verbose=verbose,
            )
            mngs.io.flush(sys=sys)
        except Exception as e:
            print(e)

<<<<<<< HEAD
    # Close open file handles
=======
>>>>>>> ff1c0838bf7bcd958d4181b20c2a42711bd4b920
    try:
        sys.stdout.close()
        sys.stderr.close()
    except:
        pass
<<<<<<< HEAD

    # RUNNING to RUNNING2FINISHEDED
    running2finished(CONFIG["SDIR"])
=======
>>>>>>> ff1c0838bf7bcd958d4181b20c2a42711bd4b920


def running2finished(src_dir, remove_src_dir=True, max_wait=60):
    dest_dir = src_dir.replace("RUNNING/", "FINISHED/")
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
        while not os.path.exists(dest_dir) and time.time() - start_time < max_wait:
            time.sleep(0.1)
        if os.path.exists(dest_dir):
            mngs.gen.print_block(
                f"Congratulations! The script completed.\n\n{dest_dir}",
                c="yellow",
            )
            if remove_src_dir:
                shutil.rmtree(src_dir)
        else:
            print(f"Copy operation timed out after {max_wait} seconds")
    except Exception as e:
        print(e)


# def running2finished(src_dir, remove_src_dir=True, max_wait=60):
#     dest_dir = src_dir.replace("RUNNING/", "FINISHED/")
#     os.makedirs(dest_dir, exist_ok=True)
#     try:
#         os.rename(src_dir, dest_dir)
#         start_time = time.time()
#         while not os.path.exists(dest_dir) and time.time() - start_time < max_wait:
#             time.sleep(0.1)
#         if os.path.exists(dest_dir):
#             mngs.gen.print_block(
#                 f"Congratulations! The script completed.\n\n{dest_dir}",
#                 c="yellow",
#             )
#             if remove_src_dir:
#                 mngs.sh(f"rm {src_dir} -rf", verbose=False)
#         else:
#             print(f"Rename operation timed out after {max_wait} seconds")
#     except Exception as e:
#         print(e)

# def running2finished(src_dir, remove_src_dir=True):
#     dest_dir = src_dir.replace("RUNNING/", "FINISHED/")
#     os.makedirs(dest_dir, exist_ok=True)
#     try:
#         os.rename(src_dir, dest_dir)
#         mngs.gen.print_block(
#             f"Congratulations! The script completed.\n\n{dest_dir}",
#             c="yellow",
#         )
#         if remove_src_dir:
#             mngs.sh(f"rm {src_dir} -rf", verbose=False)
#     except Exception as e:
#         print(e)


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
