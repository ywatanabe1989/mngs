#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-14 07:41:21 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_modules/_db.py

def _load_sqlite3db():
    if not lpath.endswith(".db"):
        raise ValueError("File must have .db extension")
    try:
        from ..db._BaseSQLiteDB import BaseSQLiteDB

        obj = BaseSQLiteDB(lpath)
        return obj
    except Exception as e:
        raise ValueError(str(e))


# EOF
