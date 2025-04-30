#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 09:22:28 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/src/mngs/types/__init__.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/types/__init__.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from typing import List, Tuple, Dict, Any, Union, Sequence, Literal, Optional, Iterable, Generator
from ._ArrayLike import ArrayLike, is_array_like
from ._is_listed_X import is_listed_X

# EOF