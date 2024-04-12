#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-12 23:27:11 (ywatanabe)"

import inspect
import os as _os
from datetime import datetime
from glob import glob
from pprint import pprint
from time import sleep

import matplotlib
import mngs


def start(
    sys=None,
    plt=None,
    sdir=None,
    verbose=True,
    # Random seeds
    os=None,
    random=None,
    np=None,
    torch=None,
    tf=None,
    seed=42,
    # matplotlib
    agg=False,
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
    alpha=0.9,
    line_width=0.5,
):
    """
    Example:

    \"""
    This script does XYZ.
    \"""

    # Imports
    import mngs
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Config
    CONFIG = mngs.gen.load_configs()

    # Functions
    # (Your awesome code here)

    if __name__ == '__main__':
        # Start
        CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

        # (Your awesome code here)

        # Close
        mngs.gen.close(CONFIG)

    # EOF

    \"""
    /home/ywatanabe/template.py
    \"""
    """

    # Timer
    start_time = datetime.now()

    # Debug mode check
    try:
        IS_DEBUG_PATH = "./config/IS_DEBUG.yaml"
        if _os.path.exists(IS_DEBUG_PATH):
            IS_DEBUG = mngs.io.load(IS_DEBUG_PATH).get("IS_DEBUG", False)
        else:
            IS_DEBUG = False

    except Exception as e:
        print(e)
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
        sdir = _sdir + sfname + "/" + "RUNNING" + "/" + ID + "/"
    _os.makedirs(sdir, exist_ok=True)

    # CONFIGs
    CONFIGS = mngs.io.load_configs(IS_DEBUG)
    CONFIGS["ID"] = ID
    CONFIGS["START_TIME"] = start_time
    CONFIGS["SDIR"] = sdir.replace("/./", "/")
    if verbose:
        print(f"\n{'-'*40}\n")
        print(f"CONFIG:")
        for k, v in CONFIGS.items():
            print(f"\n{k}:\n{v}\n")
        print(f"\n{'-'*40}\n")

    # Logging (tee)
    if sys is not None:
        sys.stdout, sys.stderr = mngs.general.tee(
            sys, sdir=sdir, verbose=verbose
        )

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
        plt, CC = mngs.plt.configure_mpl(
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
            line_width=line_width,
            verbose=verbose,
        )

    if agg:
        matplotlib.use("Agg")

    return CONFIGS, sys.stdout, sys.stderr, plt, CC


if __name__ == "__main__":
    """
    This script does XYZ.
    """

    # Imports
    import os
    import sys

    import matplotlib.pyplot as plt
    import mngs
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Config
    CONFIG = mngs.gen.load_configs()

    # Functions
    # Your awesome code here :)

    if __name__ == "__main__":
        # Start
        CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

        # Your awesome code here :)

        # Close
        mngs.gen.close(CONFIG)

# EOF

"""
/home/ywatanabe/proj/entrance/mngs/general/_start.py
"""
