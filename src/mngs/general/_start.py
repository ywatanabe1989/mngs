#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-27 19:53:28 (ywatanabe)"

import inspect
import os as _os
from datetime import datetime
from glob import glob
from pprint import pprint
from time import sleep

import mngs


def load_configs(is_debug=False):
    CONFIGS = {}
    for lpath in glob("./config/*.yaml"):
        CONFIG = mngs.io.load(lpath)
        # ???
        # fname = mngs.gen.split_fpath(lpath)[1]
        # CONFIGS[fname] = CONFIG
        # Update parameters for debugging
        if is_debug:
            debug_keys = mngs.gen.search("^DEBUG_", list(CONFIG.keys()))[1]
            for dk in debug_keys:
                dk_wo_debug_prefix = dk.split("DEBUG_")[1]
                CONFIG[dk_wo_debug_prefix] = CONFIG[dk]
                print(f"\n{dk} -> {dk_wo_debug_prefix}\n")
            sleep(1)

        CONFIGS.update(CONFIG)

    # # If only one CONFIG files are found, it should be treated as CONFIG's'
    # if len(CONFIGS) == 1:
    #     CONFIGS = CONFIGS[list(CONFIGS.keys())[0]]

    return CONFIGS


def start(
    sys=None,
    plt=None,
    sdir=None,
    show=True,
    # seeds
    os=None,
    random=None,
    np=None,
    torch=None,
    tf=None,
    seed=42,
    # matplotlib
    dpi=100,
    save_dpi=300,
    figsize=(16.2, 10),
    figscale=1.0,
    fontsize=16,
    labelsize="same",
    legendfontsize="xx-small",
    tick_size="auto",
    tick_width="auto",
    hide_spines=False,
):
    """
    import sys
    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt = mngs.gen.start(sys, plt)

    # YOUR CODE HERE

    mngs.gen.close(CONFIG)
    """
    # Timer
    start_time = datetime.now()

    # Debug mode check
    try:
        is_debug = mngs.io.load("./config/is_debug.yaml").get("DEBUG", False)
    except Exception as e:
        is_debug = False

    # ID
    ID = mngs.gen.gen_ID()
    ID = ID if not is_debug else "[DEBUG] " + ID
    print(f"\n{'#'*40}\n## {ID}\n{'#'*40}\n")
    sleep(1)

    # Defines SDIR
    if sdir is None:
        __file__ = inspect.stack()[1].filename
        if "ipython" in __file__:
            __file__ = "/tmp/fake.py"
        spath = __file__
        _sdir, sfname, _ = mngs.general.split_fpath(spath)
        sdir = _sdir + sfname + "/" + ID + "/"  # " # + "/log/"
    _os.makedirs(sdir, exist_ok=True)

    # CONFIGs
    CONFIGS = load_configs(is_debug)
    CONFIGS["ID"] = ID
    CONFIGS["START_TIME"] = start_time
    CONFIGS["SDIR"] = sdir
    if show:
        print(f"\n{'-'*40}\n")
        print(f"CONFIG:")
        for k, v in CONFIGS.items():
            print(f"\n{k}:\n{v}\n")
            sleep(0.3)
        # pprint(CONFIGS)
        print(f"\n{'-'*40}\n")

    # Logging (tee)
    if sys is not None:
        sys.stdout, sys.stderr = mngs.general.tee(sys, sdir=sdir, show=show)

    # Random seeds
    if (
        (os is not None)
        or (random is not None)
        or (np is not None)
        or (torch is not None)
        or (tf is not None)
    ):
        mngs.gen.fix_seeds(
            os=os,
            random=random,
            np=np,
            torch=torch,
            tf=tf,
            seed=seed,
            show=show,
        )

    # Matplotlib configuration
    if plt is not None:
        plt = mngs.plt.configure_mpl(
            plt,
            dpi=dpi,
            save_dpi=save_dpi,
            figsize=figsize,
            figscale=figscale,
            fontsize=fontsize,
            labelsize=labelsize,
            legendfontsize=legendfontsize,
            tick_size=tick_size,
            tick_width=tick_width,
            hide_spines=hide_spines,
            show=show,
        )

    return CONFIGS, sys.stdout, sys.stderr, plt


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import mngs

    # --------------------------------------------------------------------------- #
    CONFIG, sys.stdout, sys.stderr, plt = mngs.gen.start(sys, plt)
    # --------------------------------------------------------------------------- #

    # YOUR CODE HERE
    print("Hello world from mngs.")

    # --------------------------------------------------------------------------- #
    mngs.gen.close(CONFIG)
    # --------------------------------------------------------------------------- #

    # EOF
