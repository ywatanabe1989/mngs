#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-25 12:12:15 (ywatanabe)"

import inspect
import os as _os
from datetime import datetime
from glob import glob
from pprint import pprint
from time import sleep

import mngs


def start(
    sys=None,
    plt=None,
    sdir=None,
    show=True,
    # Random seeds
    os=None,
    random=None,
    np=None,
    torch=None,
    tf=None,
    seed=42,
    # matplotlib
    fig_size_mm=(160, 100),
    fig_scale=1.0,
    dpi_display=100,
    dpi_save=300,
    font_size_base=8,
    font_size_title=8,
    font_size_axis_label=8,
    font_size_tick_label=7,
    font_size_legend=6,
    hide_top_right_spines=True,
    alpha=0.75,
):
    """
    import sys
    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(sys, plt)

    # YOUR CODE HERE

    mngs.gen.close(CONFIG)
    """
    # Timer
    start_time = datetime.now()

    # Debug mode check
    try:
        IS_DEBUG = mngs.io.load("./config/IS_DEBUG.yaml").get(
            "IS_DEBUG", False
        )
    except Exception as e:
        IS_DEBUG = False

    # ID
    ID = mngs.gen.gen_ID(N=4)
    ID = ID if not IS_DEBUG else "DEBUG_" + ID
    print(f"\n{'#'*40}\n## {ID}\n{'#'*40}\n")
    sleep(1)

    # Defines SDIR
    if sdir is None:
        __file__ = inspect.stack()[1].filename
        if "ipython" in __file__:
            __file__ = "/tmp/fake.py"
        spath = __file__
        _sdir, sfname, _ = mngs.general.split_fpath(spath)
        sdir = (
            _sdir + sfname + "/" + "RUNNING" + "/" + ID + "/"
        )  # " # + "/log/"
    _os.makedirs(sdir, exist_ok=True)

    # CONFIGs
    CONFIGS = mngs.io.load_configs(IS_DEBUG)
    CONFIGS["ID"] = ID
    CONFIGS["START_TIME"] = start_time
    CONFIGS["SDIR"] = sdir.replace("/./", "/")
    if show:
        print(f"\n{'-'*40}\n")
        print(f"CONFIG:")
        for k, v in CONFIGS.items():
            print(f"\n{k}:\n{v}\n")
            # sleep(0.1)
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
        plt, cc = mngs.plt.configure_mpl(
            plt,
            fig_size_mm=(160, 100),
            fig_scale=fig_scale,
            dpi_display=dpi_display,
            dpi_save=dpi_save,
            font_size_base=font_size_base,
            font_size_title=font_size_title,
            font_size_axis_label=font_size_axis_label,
            font_size_tick_label=font_size_tick_label,
            font_size_legend=font_size_legend,
            hide_top_right_spines=hide_top_right_spines,
            alpha=alpha,
            show=show,
        )

    return CONFIGS, sys.stdout, sys.stderr, plt, cc


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import mngs

    # --------------------------------------------------------------------------- #
    CONFIG, sys.stdout, sys.stderr, plt, cc = mngs.gen.start(
        sys, plt, sdir=None
    )
    # --------------------------------------------------------------------------- #

    # YOUR CODE HERE
    print("Hello world from mngs.")

    # --------------------------------------------------------------------------- #
    mngs.gen.close(CONFIG)
    # --------------------------------------------------------------------------- #

    # EOF
