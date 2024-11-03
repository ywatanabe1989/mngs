#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 01:19:02 (ywatanabe)"
# File: ./mngs_repo/src/mngs/gen/_tee.py

import os as _os
import sys

from ..path import get_spath, split
from ..str._printc import printc


class Tee(object):
    """Example:
    import sys

    sys.stdout = Tee(sys.stdout, "stdout.txt")
    sys.stderr = Tee(sys.stderr, "stderr.txt")

    print("abc") # stdout
    print(1 / 0) # stderr
    # cat stdout.txt
    # cat stderr.txt
    """

    def __init__(self, sys_stdout_or_stderr, spath):
        self._files = [sys_stdout_or_stderr, open(spath, "w")]

    def __getattr__(self, attr, *args):
        return self._wrap(attr, *args)

    def _wrap(self, attr, *args):
        def g(*a, **kw):
            for f in self._files:
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
        # __file__ = inspect.stack()[1].filename
        # if "ipython" in __file__:
        #     __file__ = f"/tmp/mngs/fake_{_os.getenv('USER')}.py"
        # spath = __file__
        spath = get_spath()

        _sdir, sfname, _ = split(spath)
        sdir = _sdir + sfname

    sdir += "logs/"

    _os.makedirs(sdir, exist_ok=True)

    spath_stdout = sdir + "stdout.log"  # for printing
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
