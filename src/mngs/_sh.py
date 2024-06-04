#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-05 01:40:55 (ywatanabe)"
# Author: Yusuke Watanabe (ywata1989@gmail.com)


"""
This script does XYZ.
"""


"""
Imports
"""
import subprocess

import mngs

"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def sh(command_str, verbose=True):
    """
    Executes a shell command from Python.

    Parameters:
    - command_str (str): The command string to execute.

    Returns:
    - output (str): The standard output from the executed command.
    """
    if verbose:
        print(mngs.gen.color_text(f"{command_str}", "yellow"))

    process = subprocess.Popen(
        command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, error = process.communicate()
    if process.returncode == 0:
        out = output.decode("utf-8")
    else:
        out = error.decode("utf-8")

    if verbose:
        print(out)

    return out


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import mngs

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    shell("ls")
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
