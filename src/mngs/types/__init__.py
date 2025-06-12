#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-02 17:09:29 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/types/__init__.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/types/__init__.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# Clean up the os import to avoid namespace pollution
del os

# Clean up internal file path variables
del __FILE__, __DIR__

from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Union,
    Sequence,
    Literal,
    Optional,
    Iterable,
    Generator,
)
from ._ArrayLike import ArrayLike, is_array_like
from ._is_listed_X import is_listed_X
from ._ColorLike import ColorLike

# EOF
