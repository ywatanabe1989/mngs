#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-17 19:38:31 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_modules/_db.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/io/_load_modules/_db.py"

from typing import Any

from ...db._BaseSQLiteDB import BaseSQLiteDB


def _load_sqlite3db(lpath: str, use_temp_db: bool = False) -> Any:
    if not lpath.endswith(".db"):
        raise ValueError("File must have .db extension")
    try:
        obj = BaseSQLiteDB(lpath, use_temp_db=use_temp_db)
        return obj
    except Exception as e:
        raise ValueError(str(e))


# EOF
