#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-05-16 13:19:50 (ywatanabe)"
# Author: Yusuke Watanabe (ywata1989@gmail.com)


"""
This script does XYZ.
"""


"""
Imports
"""
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import mngs
import pandas as pd

"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def main(
    path="/tmp/mngs/processer_usages.csv",
    limit_min=3,
    interval_s=1,
    reset=True,
    verbose=True,
):
    # Parameters
    limit_s = limit_min * 60
    n_max = math.ceil(limit_s // interval_s)

    if reset and os.path.exists(path):
        mngs.sh(f"rm {path}")
        mngs.io.save(pd.DataFrame(), path, verbose=False)
        print(f"\n{path} was cleared.")

    for _ in range(n_max):
        add(path, verbose=verbose)
        time.sleep(interval_s)


def add(path, verbose=True):
    try:
        past = mngs.io.load(path)
    except Exception as e:
        print(e)
        past = pd.DataFrame()

    now = mngs.res.get_proc_usages()

    combined = pd.concat([past, now]).round(3)

    mngs.io.save(combined, path, verbose=verbose)

    if verbose:
        print(f"\n{combined}")


rec_procs = main

if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
