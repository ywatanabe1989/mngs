#!/usr/bin/env python3

"""General utility functions and classes for the MNGS project."""

# Machine Learning utilities
from ..ml.utils.grid_search import count_grids, yield_grids

# Core utilities
from .decorators.cache import cache
from .utils.ci import ci
from .utils.close import close
from .data_processing.converters import (
    batch_fn,
    numpy_fn,
    squeeze_if,
    to_numpy,
    to_torch,
    torch_fn,
    unsqueeze_if,
)
from .decorators.deprecated import deprecated
from .decorators.alternate_kwarg import alternate_kwarg
from .utils.dict_replace import dict_replace
from .data_processing.DimHandler import DimHandler
from .utils.DotDict import DotDict
from .utils.embed import embed
from .utils.less import less
from .utils.mask_api import mask_api
from .utils.not_implemented import not_implemented
from .utils.paste import paste
from .utils.src import src
from .system_ops.timeout import timeout
from .utils.title_case import title_case
from .utils.wrap import wrap

# I/O utilities
from ..io import *
from .system_ops.email import notify, send_gmail
from .system_ops.shell import run_shellcommand, run_shellscript
from .system_ops.tee import tee

# Data processing utilities
from .data_processing.norm import to_z
from .utils.reproduce import fix_seeds, gen_ID, gen_timestamp
from .utils.start import start
from .data_processing.symlog import symlog
from .utils.TimeStamper import TimeStamper
from .utils.title2path import title2path
from .utils.dict2str import dict2str
from .data_processing.transpose import transpose

# LaTeX utilities
from .utils.latex import add_hat_in_the_latex_style, to_the_latex_style

# Miscellaneous utilities
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

# Placeholder for unused variables
_ = None

