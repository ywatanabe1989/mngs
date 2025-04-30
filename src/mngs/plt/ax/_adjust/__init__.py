#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 08:47:47 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_adjust/__init__.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/ax/_adjust/__init__.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from ._add_marginal_ax import add_marginal_ax
from ._add_panel import add_panel
from ._extend import extend
from ._force_aspect import force_aspect
from ._format_label import format_label
from ._hide_spines import hide_spines
from ._map_ticks import map_ticks
from ._rotate_labels import rotate_labels
from ._sci_note import sci_note
from ._set_n_ticks import set_n_ticks
from ._set_size import set_size
from ._set_supxyt import set_supxyt
from ._set_ticks import set_ticks
from ._set_xyt import set_xyt
from ._share_axes import sharexy, sharex, sharey, get_global_xlim, get_global_ylim, set_xlims, set_ylims
from ._shift import shift

# EOF