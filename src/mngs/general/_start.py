#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-01-26 11:19:55 (ywatanabe)"

import inspect
import os as _os
from datetime import datetime
from glob import glob
from pprint import pprint

import mngs


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
    """
    # Starts timer
    start_time = datetime.now()

    ID = mngs.gen.gen_ID()

    if sdir is None:
        __file__ = inspect.stack()[1].filename
        if "ipython" in __file__:
            __file__ = "/tmp/fake.py"
        spath = __file__
        _sdir, sfname, _ = mngs.general.split_fpath(spath)
        sdir = _sdir + sfname + "/" + ID + "/"  # " # + "/log/"
    _os.makedirs(sdir, exist_ok=True)

    if sys is not None:
        sys.stdout, sys.stderr = mngs.general.tee(
            sys, sdir=sdir + "/logs/", show=show
        )

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

    # Saves as CONFIG
    CONFIGS = {}
    for lpath in glob("./config/*.yaml"):
        CONFIG = mngs.io.load(lpath)
        fname = mngs.gen.split_fpath(lpath)[1]
        CONFIGS[fname] = CONFIG
    if len(CONFIGS) == 1:
        CONFIGS = CONFIGS[list(CONFIGS.keys())[0]]
    CONFIGS["ID"] = ID
    CONFIGS["SDIR"] = sdir
    CONFIGS["START_TIME"] = start_time

    if show:
        print(f"\n{'-'*40}\n")
        print(f"CONFIG:")
        pprint(CONFIGS)
        print(f"\n{'-'*40}\n")

    return CONFIGS, sys.stdout, sys.stderr, plt


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt = mngs.gen.start(sys, plt, show=False)

    ic("aaa")
    ic("bbb")
    ic("ccc")
