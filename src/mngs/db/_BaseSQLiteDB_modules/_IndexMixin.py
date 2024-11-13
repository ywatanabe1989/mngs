#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-12 09:29:48 (ywatanabe)"
# File: ./mngs_repo/src/mngs/db/_BaseSQLiteDB_modules/_IndexMixin.py

from typing import List


class _IndexMixin:
    """Index management functionality"""

    def create_index(
        self,
        table_name: str,
        column_names: List[str],
        index_name: str = None,
        unique: bool = False,
    ) -> None:
        if index_name is None:
            index_name = f"idx_{table_name}_{'_'.join(column_names)}"
        unique_clause = "UNIQUE" if unique else ""
        query = f"CREATE {unique_clause} INDEX IF NOT EXISTS {index_name} ON {table_name} ({','.join(column_names)})"
        self.execute(query)

    def drop_index(self, index_name: str) -> None:
        self.execute(f"DROP INDEX IF EXISTS {index_name}")


# EOF
