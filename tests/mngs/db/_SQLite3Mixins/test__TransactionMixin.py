# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-29 04:32:42 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/db/_SQLite3Mixins/_TransactionMixin.py
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_SQLite3Mixins/_TransactionMixin.py"
# 
# import sqlite3
# import contextlib
# from .._BaseMixins._BaseTransactionMixin import _BaseTransactionMixin
# 
# class _TransactionMixin:
#     """Transaction management functionality"""
# 
#     @contextlib.contextmanager
#     def transaction(self):
#         with self.lock:
#             try:
#                 self.begin()
#                 yield
#                 self.commit()
#             except Exception as e:
#                 self.rollback()
#                 raise e
# 
#     def begin(self) -> None:
#         self.execute("BEGIN TRANSACTION")
# 
#     def commit(self) -> None:
#         self.conn.commit()
# 
#     def rollback(self) -> None:
#         self.conn.rollback()
# 
#     def enable_foreign_keys(self) -> None:
#         self.execute("PRAGMA foreign_keys = ON")
# 
#     def disable_foreign_keys(self) -> None:
#         self.execute("PRAGMA foreign_keys = OFF")
# 
#     @property
#     def writable(self) -> bool:
#         try:
#             self.cursor.execute("SELECT value FROM _db_state WHERE key = 'writable'")
#             result = self.cursor.fetchone()
#             return result[0].lower() == "true" if result else True
#         except sqlite3.Error:
#             return True
# 
#     @writable.setter
#     def writable(self, state: bool) -> None:
#         try:
#             self.execute("UPDATE _db_state SET protected = 0 WHERE key = 'writable'")
#             self.execute(
#                 "UPDATE _db_state SET value = ? WHERE key = 'writable'",
#                 (str(state).lower(),),
#             )
#             self.execute("UPDATE _db_state SET protected = 1 WHERE key = 'writable'")
#             self.execute("PRAGMA query_only = ?", (not state,))
#         except sqlite3.Error as err:
#             raise ValueError(f"Failed to set writable state: {err}")
# 
#     def _check_writable(self) -> None:
#         if not self.writable:
#             raise ValueError("Database is in read-only mode")
# 
# # EOF

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs..db._SQLite3Mixins._TransactionMixin import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        pass

    def test_edge_cases(self):
        # Edge case testing
        pass

    def test_error_handling(self):
        # Error handling testing
        pass
