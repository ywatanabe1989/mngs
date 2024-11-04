#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 15:56:17 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_configs.py

import os
from ..dict import DotDict
from ..io._load import load
from ._glob import glob

def load_configs(IS_DEBUG=None, show=False, verbose=False):
    """
    Load configuration files from the ./config directory.

    Parameters:
    -----------
    IS_DEBUG : bool, optional
        If True, use debug configurations. If None, check ./config/IS_DEBUG.yaml.
    show : bool, optional
        If True, display additional information during loading.
    verbose : bool, optional
        If True, print verbose output during loading.

    Returns:
    --------
    DotDict
        A dictionary-like object containing the loaded configurations.
    """

    # def apply_debug_values(config, IS_DEBUG):
    #     if IS_DEBUG:
    #         if isinstance(config, (dict, DotDict)):
    #             for key, value in list(config.items()):
    #                 try:
    #                     if key.startswith("DEBUG_"):
    #                         dk_wo_debug_prefix = key.split("DEBUG_")[1]
    #                         config[dk_wo_debug_prefix] = value
    #                         if show or verbose:
    #                             print(f"\n{key} -> {dk_wo_debug_prefix}\n")
    #                     elif isinstance(value, (dict, DotDict)):
    #                         config[key] = apply_debug_values(value, IS_DEBUG)
    #                 except Exception as e:
    #                     print(e)
    #     return config
    def apply_debug_values(config, IS_DEBUG):
        if IS_DEBUG:
            if isinstance(config, (dict, DotDict)):
                for key, value in list(config.items()):
                    try:
                        if key.startswith(("DEBUG_", "debug_")):
                            dk_wo_debug_prefix = key.split("_", 1)[1]
                            config[dk_wo_debug_prefix] = value
                            if show or verbose:
                                print(f"\n{key} -> {dk_wo_debug_prefix}\n")
                        elif isinstance(value, (dict, DotDict)):
                            config[key] = apply_debug_values(value, IS_DEBUG)
                    except Exception as e:
                        print(e)
        return config

    if os.getenv("CI") == "True":
        IS_DEBUG = True

    try:
        # Check ./config/IS_DEBUG.yaml file if IS_DEBUG argument is not passed
        if IS_DEBUG is None:
            IS_DEBUG_PATH = "./config/IS_DEBUG.yaml"
            if os.path.exists(IS_DEBUG_PATH):
                IS_DEBUG = load("./config/IS_DEBUG.yaml").get("IS_DEBUG")
            else:
                IS_DEBUG = False

        # Main
        CONFIGS = {}
        for lpath in glob("./config/*.yaml"):
            config = load(lpath)
            if config:
                CONFIG = apply_debug_values(config, IS_DEBUG)
                CONFIGS.update(CONFIG)

        CONFIGS = DotDict(CONFIGS)

    except Exception as e:
        print(e)
        CONFIGS = DotDict({})

    return CONFIGS


# EOF
