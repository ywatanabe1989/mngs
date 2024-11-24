#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-25 00:31:33 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_modules/_db.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/io/_load_modules/_db.py"

from typing import Any

from ...db._SQLite3 import SQLite3


def _load_sqlite3db(lpath: str, use_temp_db: bool = False) -> Any:
    if not lpath.endswith(".db"):
        raise ValueError("File must have .db extension")
    try:
        obj = SQLite3(lpath, use_temp=use_temp)

        return obj
    except Exception as e:
        raise ValueError(str(e))


# EOF
