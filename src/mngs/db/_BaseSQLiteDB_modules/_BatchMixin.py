#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-12 09:29:43 (ywatanabe)"
# File: ./mngs_repo/src/mngs/db/_BaseSQLiteDB_modules/_BatchMixin.py

from typing import Any as _Any
from typing import Dict, List, Optional


class _BatchMixin:
    """Batch operations functionality"""

    def _run_many(
        self,
        sql_command,
        table_name: str,
        rows: List[Dict[str, _Any]],
        batch_size: int = 1000,
        inherit_foreign: bool = True,
        where: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> None:
        assert sql_command.upper() in [
            "INSERT",
            "REPLACE",
            "INSERT OR REPLACE",
            "UPDATE",
        ]

        if not rows:
            return

        if sql_command.upper() == "UPDATE":
            valid_columns = (
                columns if columns else [col for col in rows[0].keys()]
            )
            set_clause = ",".join([f"{col}=?" for col in valid_columns])
            where_clause = where if where else "1=1"
            query = (
                f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
            )

            for idx in range(0, len(rows), batch_size):
                batch = rows[idx : idx + batch_size]
                values = [
                    tuple([row[col] for col in valid_columns]) for row in batch
                ]
                self.executemany(query, values)
            return

        if where:
            filtered_rows = []
            for row in rows:
                try:
                    test_query = f"SELECT 1 FROM (SELECT {','.join(f'{k} as {k}' for k in row.keys())}) WHERE {where}"
                    values = tuple(row.values())
                    result = self.execute(test_query, values).fetchone()
                    if result:
                        filtered_rows.append(row)
                except Exception as e:
                    print(
                        f"Warning: Where clause evaluation failed for row: {e}"
                    )
            rows = filtered_rows

        schema = self.get_table_schema(table_name)
        table_columns = set(schema["name"])
        valid_columns = [col for col in rows[0].keys()]

        if inherit_foreign:
            fk_query = f"PRAGMA foreign_key_list({table_name})"
            foreign_keys = self.execute(fk_query).fetchall()

            for row in rows:
                for fk in foreign_keys:
                    ref_table, from_col, to_col = fk[2], fk[3], fk[4]
                    if from_col not in row or row[from_col] is None:
                        if to_col in row:
                            query = f"SELECT {from_col} FROM {ref_table} WHERE {to_col} = ?"
                            result = self.execute(
                                query, (row[to_col],)
                            ).fetchone()
                            if result:
                                row[from_col] = result[0]

        columns = valid_columns
        placeholders = ",".join(["?" for _ in columns])
        query = f"{sql_command} INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"

        for idx in range(0, len(rows), batch_size):
            batch = rows[idx : idx + batch_size]
            values = [[row.get(col) for col in valid_columns] for row in batch]
            self.executemany(query, values)

    def update_many(
        self,
        table_name: str,
        rows: List[Dict[str, _Any]],
        batch_size: int = 1000,
        where: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> None:
        with self.transaction():
            self._run_many(
                sql_command="UPDATE",
                table_name=table_name,
                rows=rows,
                batch_size=batch_size,
                inherit_foreign=False,
                where=where,
                columns=columns,
            )

    def insert_many(
        self,
        table_name: str,
        rows: List[Dict[str, _Any]],
        batch_size: int = 1000,
        inherit_foreign: bool = True,
        where: Optional[str] = None,
    ) -> None:
        with self.transaction():
            self._run_many(
                sql_command="INSERT",
                table_name=table_name,
                rows=rows,
                batch_size=batch_size,
                inherit_foreign=inherit_foreign,
                where=where,
            )

    def replace_many(
        self,
        table_name: str,
        rows: List[Dict[str, _Any]],
        batch_size: int = 1000,
        inherit_foreign: bool = True,
        where: Optional[str] = None,
    ) -> None:
        with self.transaction():
            self._run_many(
                sql_command="REPLACE",
                table_name=table_name,
                rows=rows,
                batch_size=batch_size,
                inherit_foreign=inherit_foreign,
                where=where,
            )

    def delete_where(
        self, table_name: str, where: str, limit: Optional[int] = None
    ) -> None:
        with self.transaction():
            query = f"DELETE FROM {table_name} WHERE {where}"
            if limit is not None:
                query += f" LIMIT {limit}"
            self.execute(query)

    def update_where(
        self,
        table_name: str,
        updates: Dict[str, _Any],
        where: str,
        limit: Optional[int] = None,
    ) -> None:
        with self.transaction():
            set_clause = ", ".join([f"{col} = ?" for col in updates.keys()])
            query = f"UPDATE {table_name} SET {set_clause} WHERE {where}"
            if limit is not None:
                query += f" LIMIT {limit}"
            self.execute(query, tuple(updates.values()))


# EOF
