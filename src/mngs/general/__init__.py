#!/usr/bin/env python3


from ._close import close
from ._converters import (
    numpy_fn,
    squeeze_if,
    to_numpy,
    to_torch,
    torch_fn,
    unsqueeze_if,
)
from ._embed import embed

# Confirmed
from ._paste import paste
from ._reload import reload

_ = None
from ..io.__init__ import *
from ._cuda_collect_env import main as cuda_collect_env
from ._norm import to_z
from ._reproduce import fix_seeds, gen_ID, gen_timestamp, tee
from ._shell import run_shellcommand, run_shellscript
from ._start import start
from ._TimeStamper import TimeStamper
from .email import notify, send_gmail
from .latex import add_hat_in_the_latex_style, to_the_latex_style

# from .mat2py import *
from .misc import (
    _return_counting_process,
    color_text,
    connect_nums,
    connect_strs,
    copy_files,
    copy_the_file,
    ct,
    decapitalize,
    describe,
    fmt_size,
    grep,
    is_defined_global,
    is_defined_local,
    is_later_or_equal,
    is_listed_X,
    is_nan,
    isclose,
    listed_dict,
    merge_dicts_wo_overlaps,
    mv_col,
    partial_at,
    pop_keys,
    print_block,
    search,
    squeeze_spaces,
    suppress_output,
    symlink,
    take_the_closest,
    to_even,
    to_odd,
    unique,
    uq,
    wait_key,
)
from .pandas import (
    col_to_last,
    col_to_top,
    force_dataframe,
    ignore_SettingWithCopyWarning,
    merge_columns,
)
