#!/usr/bin/env python3

import fnmatch
import glob
import inspect
import os
import re
import subprocess
import warnings
from glob import glob

import mngs

if "general" in __file__:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(
            '\n"mngs.general.path" will be removed. '
            'Please use "mngs.io.path" instead.',
            PendingDeprecationWarning,
        )


################################################################################
## PATH
################################################################################
def get_this_fpath(when_ipython="/tmp/fake.py"):
    __file__ = inspect.stack()[1].filename
    if "ipython" in __file__:  # for ipython
        __file__ = when_ipython  # "/tmp/fake.py"
    return __file__


def mk_spath(sfname, makedirs=False):

    __file__ = inspect.stack()[1].filename
    if "ipython" in __file__:  # for ipython
        __file__ = f'/tmp/fake-{os.getenv("USER")}.py'

    # __file__ = get_current_file_name()

    ## spath
    fpath = __file__
    fdir, fname, _ = split_fpath(fpath)
    sdir = fdir + fname + "/"
    spath = sdir + sfname

    if makedirs:
        os.makedirs(mngs.general.split_fpath(spath)[0], exist_ok=True)

    return spath


def find_the_git_root_dir():
    import git

    repo = git.Repo(".", search_parent_directories=True)
    return repo.working_tree_dir


def split_fpath(fpath):
    """Split a file path to (1) the directory path, (2) the file name, and (3) the file extention
    Example:
        dirname, fname, ext = split_fpath('../data/01/day1/split_octave/2kHz_mat/tt8-2.mat')
        print(dirname) # '../data/01/day1/split_octave/2kHz_mat/'
        print(fname) # 'tt8-2'
        print(ext) # '.mat'
    """
    dirname = os.path.dirname(fpath) + "/"
    base = os.path.basename(fpath)
    fname, ext = os.path.splitext(base)
    return dirname, fname, ext


def touch(fpath):
    import pathlib

    return pathlib.Path(fpath).touch()


def find(rootdir, type="f", exp=["*"]):
    """
    Mimicks the Unix find command.

    Example:
        # rootdir =
        # type = 'f'  # 'f' for files, 'd' for directories, None for both
        # exp = '*.txt'  # Pattern to match, or None to match all
        find('/path/to/search', "f", "*.txt")
    """
    if isinstance(exp, str):
        exp = [exp]

    matches = []
    for _exp in exp:
        for root, dirs, files in os.walk(rootdir):
            # Depending on the type, choose the list to iterate over
            if type == "f":  # Files only
                names = files
            elif type == "d":  # Directories only
                names = dirs
            else:  # All entries
                names = files + dirs

            for name in names:
                # Construct the full path
                path = os.path.join(root, name)

                # If an _exp is provided, use fnmatch to filter names
                if _exp and not fnmatch.fnmatch(name, _exp):
                    continue

                # If type is set, ensure the type matches
                if type == "f" and not os.path.isfile(path):
                    continue
                if type == "d" and not os.path.isdir(path):
                    continue

                # Add the matching path to the results
                matches.append(path)

    return matches


def find_latest(dirname, fname, ext, version_prefix="_v"):
    version_pattern = re.compile(
        rf"({re.escape(fname)}{re.escape(version_prefix)})(\d+)({re.escape(ext)})$"
    )

    glob_pattern = os.path.join(dirname, f"{fname}{version_prefix}*{ext}")
    files = glob(glob_pattern)

    highest_version = 0
    latest_file = None

    for file in files:
        filename = os.path.basename(file)
        match = version_pattern.search(filename)
        if match:
            version_num = int(match.group(2))
            if version_num > highest_version:
                highest_version = version_num
                latest_file = file

    return latest_file


def increment_version(dirname, fname, ext, version_prefix="_v"):
    # Create a regex pattern to match the version number in the filename
    version_pattern = re.compile(
        rf"({re.escape(fname)}{re.escape(version_prefix)})(\d+)({re.escape(ext)})$"
    )

    # Construct the glob pattern to find all files that match the pattern
    glob_pattern = os.path.join(dirname, f"{fname}{version_prefix}*{ext}")

    # Use glob to find all files that match the pattern
    files = glob(glob_pattern)

    # Initialize the highest version number
    highest_version = 0
    base, suffix = None, None

    # Loop through the files to find the highest version number
    for file in files:
        filename = os.path.basename(file)
        match = version_pattern.search(filename)
        if match:
            base, version_str, suffix = match.groups()
            version_num = int(version_str)
            if version_num > highest_version:
                highest_version = version_num

    # If no versioned files were found, use the provided filename and extension
    if base is None or suffix is None:
        base = f"{fname}{version_prefix}"
        suffix = ext
        highest_version = 0  # No previous versions

    # Increment the highest version number
    next_version_number = highest_version + 1

    # Format the next version number with the same number of digits as the original
    next_version_str = f"{base}{next_version_number:03d}{suffix}"

    # Combine the directory and new filename to create the full path
    next_filepath = os.path.join(dirname, next_version_str)

    return next_filepath
