#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-24 22:30:57 (ywatanabe)"
# File: ./mngs_repo/src/mngs/db/_PostgreSQL_modules/_TransactionMixin.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_PostgreSQL_modules/_TransactionMixin.py"

import psycopg2
from .._BaseMixins._BaseTransactionMixin import _BaseTransactionMixin

class _TransactionMixin(_BaseTransactionMixin):
    def begin(self) -> None:
        self.execute("BEGIN TRANSACTION")

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def enable_foreign_keys(self) -> None:
        # In PostgreSQL, foreign key constraints are always enabled
        pass

    def disable_foreign_keys(self) -> None:
        # Warning: This is session-level and should be used carefully
        self.execute("SET session_replication_role = 'replica'")

    @property
    def writable(self) -> bool:
        try:
            self.cursor.execute(
                "SELECT current_setting('transaction_read_only') = 'off'"
            )
            return self.cursor.fetchone()[0]
        except psycopg2.Error:
            return True

    @writable.setter
    def writable(self, state: bool) -> None:
        try:
            if state:
                self.execute("SET TRANSACTION READ WRITE")
            else:
                self.execute("SET TRANSACTION READ ONLY")
        except psycopg2.Error as err:
            raise ValueError(f"Failed to set writable state: {err}")

    def _check_writable(self) -> None:
        if not self.writable:
            raise ValueError("Database is in read-only mode")


# EOF
