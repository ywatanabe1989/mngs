#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-07 19:18:35 (ywatanabe)"
# File: ./mngs_repo/tests/custom/test_mngs_run.py

import mngs

"""
1. Functionality:
   - (e.g., Executes XYZ operation)
2. Input:
   - (e.g., Required data for XYZ)
3. Output:
   - (e.g., Results of XYZ operation)
4. Prerequisites:
   - (e.g., Necessary dependencies for XYZ)

(Remove me: Please fill docstrings above, while keeping the bulette point style, and remove this instruction line)
"""

"""Imports"""
"""Warnings"""
# mngs.pd.ignore_SettingWithCopyWarning()
# warnings.simplefilter("ignore", UserWarning)
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", UserWarning)

"""Parameters"""
# from mngs.io import load_configs
# CONFIG = load_configs()

"""Functions & Classes"""
def main():
    return 0

def test_main():
    # -----------------------------------
    # Initiatialization of mngs format
    # -----------------------------------
    import sys

    import matplotlib.pyplot as plt

    # Configurations
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
        # sdir_suffix="",
    )

    # # Argument parser
    # script_mode = mngs.gen.is_script()
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, choices=None, default=1, help='(default: %%(default)s)')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='(default: %%(default)s)')
    # args = parser.parse_args()
    # mngs.gen.print_block(args, c='yellow')

    # -----------------------------------
    # Main
    # -----------------------------------
    exit_status = main()

    # -----------------------------------
    # Cleanup mngs format
    # -----------------------------------
    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )

    assert True

if __name__ == '__main__':
    test_main()


# EOF
