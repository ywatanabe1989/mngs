#!/usr/bin/env python3

from ..ml.utils.grid_search import count_grids, yield_grids
from ._cache import cache
from ._ci import ci
from ._close import close
from ._converters import (
    batch_fn,
    numpy_fn,
    squeeze_if,
    to_numpy,
    to_torch,
    torch_fn,
    unsqueeze_if,
)
from ._deprecated import deprecated
from ._dict_replace import dict_replace
from ._DimHandler import DimHandler

# from ._ddict import ddict
from ._DotDict import DotDict
from ._embed import embed

# from ._find_indi import find_indi
from ._less import less
from ._mask_api import mask_api
from ._not_implemented import not_implemented

# Confirmed
from ._paste import paste
from ._src import src
from ._timeout import timeout
from ._title_case import title_case
from ._wrap import wrap

_ = None
from ..io.__init__ import *
from ._email import notify, send_gmail
from ._norm import to_z
from ._reproduce import fix_seeds, gen_ID, gen_timestamp
from ._shell import run_shellcommand, run_shellscript
from ._start import start
from ._symlog import symlog
from ._tee import tee
from ._TimeStamper import TimeStamper
from ._title2path import title2path
from .latex import add_hat_in_the_latex_style, to_the_latex_style
from .misc import (
    _return_counting_process,
    color_text,  # mv_col,
    connect_nums,
    connect_strs,
    copy_files,
    copy_the_file,
    ct,
    decapitalize,
    describe,
    find_closest,
    float_linspace,
    grep,
    is_defined_global,
    is_defined_local,
    is_later_or_equal,
    is_listed_X,
    is_nan,
    isclose,
    listed_dict,
    merge_dicts_wo_overlaps,
    natglob,
    partial_at,
    pop_keys,
    print_block,
    quiet,
    readable_bytes,
    replace,
    search,
    squeeze_spaces,
    suppress_output,
    symlink,
    to_even,
    to_odd,
    unique,
    uq,
    wait_key,
)
