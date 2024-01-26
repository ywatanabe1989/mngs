#!/usr/bin/env python3

# from .path import find_the_git_root_dir, get_this_fpath, mk_spath, split_fpath
# from .load import get_data_path_from_a_package, load
# from .save import is_listed_X, save, save_listed_dfs_as_csv, save_listed_scalars_as_csv
from ..io.__init__ import *
from ._close import close
from ._start import start
from .calc import to_z
from .converters import (
    KeepPD,
    my2array,
    my2tensor,
    three2two_dim,
    two2three_dim,
)
from .cuda_collect_env import main as cuda_collect_env
from .debug import *
from .email import notify, send_gmail
from .latex import add_hat_in_the_latex_style, to_the_latex_style
from .mat2py import *
from .misc import (
    _return_counting_process,
    connect_nums,
    connect_strs,
    copy_files,
    copy_the_file,
    decapitalize,
    describe,
    fmt_size,
    grep,
    is_defined_global,
    is_defined_local,
    is_later_or_equal,
    is_nan,
    isclose,
    listed_dict,
    merge_dicts_wo_overlaps,
    partial_at,
    pop_keys,
    search,
    squeeze_spaces,
    suppress_output,
    take_the_closest,
    wait_key,
)
from .pandas import (
    col_to_last,
    col_to_top,
    force_dataframe,
    ignore_SettingWithCopyWarning,
    merge_columns,
)
from .repro import *
from .TimeStamper import *
from .torch import *
