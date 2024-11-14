#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-15 02:43:38 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_modules/_db.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/io/_load_modules/_db.py"


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-14 07:57:13 (ywatanabe)"
# File: ./mngs_repo/src/mngs/io/_load_modules/_db.py

from typing import Any
from ...db._BaseSQLiteDB import BaseSQLiteDB

def _load_sqlite3db(lpath: str, use_temp: bool = False, **kwargs) -> Any:
    if not lpath.endswith(".db"):
        raise ValueError("File must have .db extension")
    try:
        obj = BaseSQLiteDB(lpath, use_temp=use_temp)
        return obj
    except Exception as e:
        raise ValueError(str(e))


# EOF
