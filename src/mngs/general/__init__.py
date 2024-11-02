#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 03:53:14 (ywatanabe)"
# File: ./mngs_repo/src/mngs/general/__init__.py
#!/usr/bin/env python3

"""General utility functions and classes for the MNGS project."""

# I/O utilities
from ..io import *

# Machine Learning utilities
from ..ml.utils.grid_search import count_grids, yield_grids
from ._is_ipython import is_ipython, is_script
#
from ._converters import (
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

from ._suppress_output import suppress_output, quiet
# Data processing utilities
from ._ci import ci
from ._DimHandler import DimHandler
from ._norm import to_z, to_nanz
from ._symlog import symlog
from ._to_rank import to_rank
from ._transpose import transpose
from ._alternate_kwarg import alternate_kwarg

# Core utilities
from ._cache import cache
from ._deprecated import deprecated

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
    readable_bytes,
    replace,
    search,
    squeeze_spaces,
    symlink,
    to_even,
    to_odd,
    unique,
    uq,
    wait_key,
)
from ._parse_str import parse_str
# Utils
from ._email import send_gmail
from ._notify import notify
from ._shell import run_shellcommand, run_shellscript
from ._tee import tee
from ._timeout import timeout
from ._close import close
from ._dict2str import dict2str
from ._dict_replace import dict_replace
from ._DotDict import DotDict
from ._embed import embed

# LaTeX utilities
from ._latex import add_hat_in_latex_style, to_latex_style
from ._less import less
from ._mask_api import mask_api
from ._not_implemented import not_implemented
from ._paste import paste
from ._reproduce import fix_seeds, gen_ID, gen_timestamp
from ._src import src
from ._start import start
from ._TimeStamper import TimeStamper
from ._title2path import title2path
from ._title_case import title_case
from ._wrap import wrap

from ._inspect_module import inspect_module

# Placeholder for unused variables
_ = None


# EOF
