#!/usr/bin/env python3


import inspect
import os

import mngs


def this_path(when_ipython="/tmp/fake.py"):
    __file__ = inspect.stack()[1].filename
    if "ipython" in __file__:  # for ipython
        __file__ = when_ipython  # "/tmp/fake.py"
    return __file__


get_this_path = this_path


def spath(sfname=".", makedirs=False):

    __file__ = inspect.stack()[1].filename
    if "ipython" in __file__:  # for ipython
        __file__ = f'/tmp/fake-{os.getenv("USER")}.py'

    ## spath
    fpath = __file__
    fdir, fname, _ = split(fpath)
    sdir = fdir + fname + "/"
    spath = sdir + sfname

    if makedirs:
        os.makedirs(mngs.path.split(spath)[0], exist_ok=True)

    return spath


mk_spath = spath


def split(fpath):
    """Split a file path to (1) the directory path, (2) the file name, and (3) the file extention
    Example:
        dirname, fname, ext = split('../data/01/day1/split_octave/2kHz_mat/tt8-2.mat')
        print(dirname) # '../data/01/day1/split_octave/2kHz_mat/'
        print(fname) # 'tt8-2'
        print(ext) # '.mat'
    """
    dirname = os.path.dirname(fpath) + "/"
    base = os.path.basename(fpath)
    fname, ext = os.path.splitext(base)
    return dirname, fname, ext


def file_size(path):
    if os.path.exists(path):
        file_size_bytes = os.path.getsize(path)
        return mngs.gen.readable_bytes(file_size_bytes)
    else:
        return "(Not Found)"
