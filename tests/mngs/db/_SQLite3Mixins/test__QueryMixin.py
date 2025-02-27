# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-29 04:31:43 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/db/_SQLite3Mixins/_QueryMixin.py
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_SQLite3Mixins/_QueryMixin.py"
# 
# import sqlite3
# from typing import List, Tuple
# 
# import pandas as pd
# from .._BaseMixins._BaseQueryMixin import _BaseQueryMixin
# 
# class _QueryMixin:
#     """Query execution functionality"""
# 
#     def _sanitize_parameters(self, parameters):
#         """Convert pandas Timestamp objects to strings"""
#         if isinstance(parameters, (list, tuple)):
#             return [str(p) if isinstance(p, pd.Timestamp) else p for p in parameters]
#         return parameters
# 
#     def execute(self, query: str, parameters: Tuple = ()) -> None:
#         if not self.cursor:
#             raise ConnectionError("Database not connected")
# 
#         if any(keyword in query.upper()
#                for keyword in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]):
#             self._check_writable()
# 
#         try:
#             parameters = self._sanitize_parameters(parameters)
#             self.cursor.execute(query, parameters)
#             self.conn.commit()
#             return self.cursor
#         except sqlite3.Error as err:
#             raise sqlite3.Error(f"Query execution failed: {err}")
# 
#     def executemany(self, query: str, parameters: List[Tuple]) -> None:
#         if not self.cursor:
#             raise ConnectionError("Database not connected")
# 
#         if any(keyword in query.upper()
#                for keyword in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]):
#             self._check_writable()
# 
#         try:
#             parameters = [self._sanitize_parameters(p) for p in parameters]
#             self.cursor.executemany(query, parameters)
#             self.conn.commit()
#         except sqlite3.Error as err:
#             raise sqlite3.Error(f"Batch query execution failed: {err}")
# 
#     def executescript(self, script: str) -> None:
#         if not self.cursor:
#             raise ConnectionError("Database not connected")
# 
#         if any(
#             keyword in script.upper()
#             for keyword in [
#                 "INSERT",
#                 "UPDATE",
#                 "DELETE",
#                 "DROP",
#                 "CREATE",
#                 "ALTER",
#             ]
#         ):
#             self._check_writable()
# 
#         try:
#             self.cursor.executescript(script)
#             self.conn.commit()
#         except sqlite3.Error as err:
#             raise sqlite3.Error(f"Script execution failed: {err}")
# 
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

from ...src.mngs..db._SQLite3Mixins._QueryMixin import *

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
