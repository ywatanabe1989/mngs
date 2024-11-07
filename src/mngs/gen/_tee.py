#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 00:26:35 (ywatanabe)"
# File: ./mngs_repo/src/mngs/gen/_tee.py

import os as _os
import re
import sys

from ..path import split
from ..str._printc import printc


class Tee(object):
    """Example:
    import sys
    sys.stdout = Tee(sys.stdout, "stdout.txt")
    sys.stderr = Tee(sys.stderr, "stderr.txt")
    """
    def __init__(self, sys_stdout_or_stderr, spath):
        self._files = [sys_stdout_or_stderr, open(spath, "w")]
        self._is_stderr = sys_stdout_or_stderr is sys.stderr

    def __getattr__(self, attr, *args):
        return self._wrap(attr, *args)

    def _wrap(self, attr, *args):
        def g(*a, **kw):
            for f in self._files:
                if self._is_stderr and f is not sys.stderr:
                    # Filter tqdm lines from log file
                    msg = a[0] if a else ""
                    if not re.match(r'^[\s]*[0-9]+%.*\[A*$', msg):
                        res = getattr(f, attr, *args)(*a, **kw)
                else:
                    res = getattr(f, attr, *args)(*a, **kw)
            return res
        return g

def tee(sys, sdir=None, verbose=True):
    """
    import sys

    sys.stdout, sys.stderr = tee(sys)

    print("abc")  # stdout
    print(1 / 0)  # stderr
    """

    import inspect

    ####################
    ## Determines sdir
    ####################
    if sdir is None:
        __file__ = inspect.stack()[1].filename
        if "ipython" in __file__:
            __file__ = f"/tmp/mngs/{_os.getenv('USER')}.py"
        spath = __file__
        _sdir, sfname, _ = split(spath)
        sdir = _sdir + sfname

    sdir += "logs/"

    _os.makedirs(sdir, exist_ok=True)

    spath_stdout = sdir + "stdout.log"
    spath_stderr = sdir + "stderr.log"
    sys_stdout = Tee(sys.stdout, spath_stdout)
    sys_stderr = Tee(sys.stdout, spath_stderr)

    if verbose:
        message = f"Standard output/error are being logged at:\n\t{spath_stdout}\n\t{spath_stderr}"
        printc(message)

    return sys_stdout, sys_stderr

main = tee

if __name__ == "__main__":
    # # Argument Parser
    import matplotlib.pyplot as plt
    import mngs

    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()
    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main(sys, CONFIG["SDIR"])
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
