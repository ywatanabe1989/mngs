#!/usr/bin/env python3

"""General utility functions and classes for the MNGS project."""

# I/O utilities
from ..io import *

# Machine Learning utilities
from ..ml.utils.grid_search import count_grids, yield_grids

#
from .decorators._converters import (
    batch_fn,
    numpy_fn,
    squeeze_if,
    to_numpy,
    to_torch,
    torch_fn,
    pandas_fn,
    unsqueeze_if,
)

None # to keep order when black is applied

# Data processing utilities
from .data_processing._ci import ci
from .data_processing._DimHandler import DimHandler
from .data_processing._norm import to_z
from .data_processing._symlog import symlog
from .data_processing._to_rank import to_rank
from .data_processing._transpose import transpose
from .decorators._alternate_kwarg import alternate_kwarg

# Core utilities
from .decorators._cache import cache
from .decorators._deprecated import deprecated

# Miscellaneous utilities
from .misc import (
    _return_counting_process,
    color_text,
    connect_nums,
    connect_strs,
    copy_files,
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
    print_,
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

# Utils
from .system_ops._email import send_gmail
from .system_ops._notify import notify
from .system_ops._shell import run_shellcommand, run_shellscript
from .system_ops._tee import tee
from .system_ops._timeout import timeout
from .utils._close import close
from .utils._dict2str import dict2str
from .utils._dict_replace import dict_replace
from .utils._DotDict import DotDict
from .utils._embed import embed

# LaTeX utilities
from .utils._latex import add_hat_in_latex_style, to_latex_style
from .utils._less import less
from .utils._mask_api import mask_api
from .utils._not_implemented import not_implemented
from .utils._paste import paste
from .utils._reproduce import fix_seeds, gen_ID, gen_timestamp
from .utils._src import src
from .utils._start import start
from .utils._TimeStamper import TimeStamper
from .utils._title2path import title2path
from .utils._title_case import title_case
from .utils._wrap import wrap

from ._inspect_module import inspect_module

# Placeholder for unused variables
_ = None
