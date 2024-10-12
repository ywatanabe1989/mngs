#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Time-stamp: "2024-10-10 18:35:43 (ywatanabe)"
=======
# Time-stamp: "2024-10-10 19:56:30 (ywatanabe)"
>>>>>>> a3af025d0839ee245ab423e7ceac389669881898

import inspect
import os as _os
import re
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
    font_size_base=10,
    font_size_title=10,
    font_size_axis_label=10,
    font_size_tick_label=8,
    font_size_legend=8,
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

    # Initialize plt
    plt.close("all")

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
    PID = _os.getpid()
    print(
        f"\n{'#'*40}\n## mngs v{mngs.__version__}\n## {ID} (PID: {PID})\n{'#'*40}\n"
    )
    sleep(1)

    # Defines SDIR
    if sdir is None:
        __file__ = inspect.stack()[1].filename
        if "ipython" in __file__:
            __file__ = f"/tmp/fake_{_os.getenv('USER')}.py"
        spath = __file__
        _sdir, sfname, _ = mngs.path.split(spath)
        sdir = _sdir + sfname + "/" + "RUNNING" + "/" + ID + "/"
        sdir = sdir.replace("/./", "/")
    _os.makedirs(sdir, exist_ok=True)

    relative_sdir = simplify_relative_path(sdir)

    # # Relative SDIR
    # base_path = _os.getcwd()
    # relative_sdir = _os.path.relpath(sdir, base_path) if base_path else sdir
    # # relative_srid = "scripts/memory-load/distance_between_gs_stats/RUNNING/2024Y-09M-12D-02h44m40s_GlBZ"
    # relative_sdir.replace("scripts/", "./").replace("RUNNING/", "").replace(2024Y-09M-12D-02h44m40s_GlBZ, "")

    # CONFIGs
    CONFIGS = mngs.io.load_configs(IS_DEBUG).to_dict()
    CONFIGS["ID"] = ID
    CONFIGS["START_TIME"] = start_time
    CONFIGS["SDIR"] = sdir
    CONFIGS["REL_SDIR"] = relative_sdir
<<<<<<< HEAD
=======
    if verbose:
        print(f"\n{'-'*40}\n")
        pprint(CONFIGS)
        # print(f"CONFIG:")
        # for k, v in CONFIGS.items():
        #     print(f"\n{k}:\n{v}\n")
        print(f"\n{'-'*40}\n")
>>>>>>> a3af025d0839ee245ab423e7ceac389669881898

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
            # tf=tf,
            seed=seed,
            # verbose=verbose,
        )

    # Matplotlib configuration
    if plt is not None:
        plt, CC = mngs.plt.configure_mpl(
            plt,
            fig_size_mm=(160, 100),
            fig_scale=fig_scale,
            dpi_display=dpi_display,
            dpi_save=dpi_save,
            # font_size_base=font_size_base,
            # font_size_title=font_size_title,
            # font_size_axis_label=font_size_axis_label,
            # font_size_tick_label=font_size_tick_label,
            # font_size_legend=font_size_legend,
            hide_top_right_spines=hide_top_right_spines,
            alpha=alpha,
            line_width=line_width,
            verbose=verbose,
        )
        CC["gray"] = CC["grey"]

    if agg:
        matplotlib.use("Agg")

    CONFIGS = mngs.gen.DotDict(CONFIGS)

    if verbose:
        print(f"\n{'-'*40}\n")
        pprint(CONFIGS)
        # for k, v in CONFIGS.items():
        #     print(f"\n{k}:\n{v}\n")
        print(f"\n{'-'*40}\n")

    return CONFIGS, sys.stdout, sys.stderr, plt, CC


def simplify_relative_path(sdir):
    """
    Simplify the relative path by removing specific patterns.

    Example
    -------
    sdir = '/home/user/scripts/memory-load/distance_between_gs_stats/RUNNING/2024Y-09M-12D-02h44m40s_GlBZ'
    simplified_path = simplify_relative_path(sdir)
    print(simplified_path)
    # Output: './memory-load/distance_between_gs_stats/'

    Parameters
    ----------
    sdir : str
        The directory path to simplify

    Returns
    -------
    str
        Simplified relative path
    """
    base_path = _os.getcwd()
    relative_sdir = _os.path.relpath(sdir, base_path) if base_path else sdir
    simplified_path = relative_sdir.replace("scripts/", "./").replace(
        "RUNNING/", ""
    )
    # Remove date-time pattern and random string
    simplified_path = re.sub(
        r"\d{4}Y-\d{2}M-\d{2}D-\d{2}h\d{2}m\d{2}s_\w+/?$", "", simplified_path
    )
    return simplified_path


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
        CONFIG, sys.stdout, sys.stderr, plt, CC = start(sys, plt)

        # Your awesome code here :)

        # Close
        mngs.gen.close(CONFIG)

# EOF

"""
/home/ywatanabe/proj/entrance/mngs/general/_start.py
"""
