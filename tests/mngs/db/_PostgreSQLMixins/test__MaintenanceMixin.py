# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-24 23:02:38 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/db/_PostgreSQLMixins/_MaintenanceMixin.py
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_PostgreSQLMixins/_MaintenanceMixin.py"
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-24 23:00:04 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/db/_PostgreSQLMixins/_MaintenanceMixin.py
# 
# import pandas as pd
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_PostgreSQLMixins/_MaintenanceMixin.py"
# 
# import contextlib
# from typing import ContextManager, Dict, List, Optional
# 
# import psycopg2
# 
# from .._BaseMixins._BaseMaintenanceMixin import _BaseMaintenanceMixin
# 
# 
# class _MaintenanceMixin(_BaseMaintenanceMixin):
#     @contextlib.contextmanager
#     def maintenance_lock(self) -> ContextManager[None]:
#         if not self._maintenance_lock.acquire(timeout=300):
#             raise TimeoutError("Could not acquire maintenance lock")
#         try:
#             yield
#         finally:
#             self._maintenance_lock.release()
# 
#     def vacuum(self, table: Optional[str] = None, full: bool = False) -> None:
#         """Execute VACUUM on the specified table or entire database"""
#         with self.maintenance_lock():
#             try:
#                 self._check_writable()
#                 query = "VACUUM"
#                 if full:
#                     query += " FULL"
#                 if table:
#                     query += f" {table}"
#                 self.execute(query)
#             except psycopg2.Error as err:
#                 raise ValueError(f"Vacuum operation failed: {err}")
# 
#     def analyze(self, table: Optional[str] = None) -> None:
#         """Update statistics for query optimization"""
#         with self.maintenance_lock():
#             try:
#                 self._check_writable()
#                 query = "ANALYZE"
#                 if table:
#                     query += f" {table}"
#                 self.execute(query)
#             except psycopg2.Error as err:
#                 raise ValueError(f"Analyze operation failed: {err}")
# 
#     def reindex(self, table: Optional[str] = None) -> None:
#         """Rebuild indexes"""
#         with self.maintenance_lock():
#             try:
#                 self._check_writable()
#                 if table:
#                     self.execute(f"REINDEX TABLE {table}")
#                 else:
#                     self.execute("REINDEX DATABASE CURRENT_DATABASE()")
#             except psycopg2.Error as err:
#                 raise ValueError(f"Reindex operation failed: {err}")
# 
#     def get_table_size(self, table: str) -> str:
#         """Get the size of a specific table"""
#         try:
#             query = """
#                 SELECT pg_size_pretty(pg_total_relation_size(%s))
#             """
#             self.execute(query, (table,))
#             return self.cursor.fetchone()[0]
#         except psycopg2.Error as err:
#             raise ValueError(f"Failed to get table size: {err}")
# 
#     def get_database_size(self) -> str:
#         """Get the size of the current database"""
#         try:
#             query = """
#                 SELECT pg_size_pretty(pg_database_size(current_database()))
#             """
#             self.execute(query)
#             return self.cursor.fetchone()[0]
#         except psycopg2.Error as err:
#             raise ValueError(f"Failed to get database size: {err}")
# 
#     def get_table_info(self) -> List[Dict]:
#         """Get information about all tables in the database"""
#         try:
#             query = """
#                 SELECT
#                     table_name,
#                     pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size,
#                     (SELECT count(*) FROM information_schema.columns
#                      WHERE table_name=tables.table_name) as columns,
#                     (SELECT COUNT(*) FROM information_schema.table_constraints
#                      WHERE table_name=tables.table_name AND constraint_type='PRIMARY KEY') as has_pk
#                 FROM information_schema.tables
#                 WHERE table_schema = 'public'
#                 ORDER BY table_name;
#             """
#             self.execute(query)
#             columns = [desc[0] for desc in self.cursor.description]
#             return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
#         except psycopg2.Error as err:
#             raise ValueError(f"Failed to get table information: {err}")
# 
#     def optimize(self, table: Optional[str] = None) -> None:
#         """Perform full optimization on table or database"""
#         with self.maintenance_lock():
#             try:
#                 self.vacuum(table, full=True)
#                 self.analyze(table)
#                 self.reindex(table)
#             except ValueError as err:
#                 raise ValueError(f"Optimization failed: {err}")
# 
#     def get_summaries(
#         self,
#         table_names: Optional[List[str]] = None,
#         verbose: bool = True,
#         limit: int = 5,
#     ) -> Dict[str, pd.DataFrame]:
# 
#         try:
#             if table_names is None:
#                 table_names = self.get_table_names()
#             if isinstance(table_names, str):
#                 table_names = [table_names]
# 
#             sample_tables = {}
#             for table_name in table_names:
#                 query = f"""
#                     SELECT *
#                     FROM {table_name}
#                     LIMIT {limit}
#                 """
#                 self.execute(query)
#                 columns = [desc[0] for desc in self.cursor.description]
#                 rows = self.cursor.fetchall()
#                 table_sample = pd.DataFrame(rows, columns=columns)
# 
#                 for column in table_sample.columns:
#                     if table_sample[column].dtype == object:
#                         try:
#                             pd.to_datetime(table_sample[column], format='mixed', errors='raise')
#                             continue
#                         except (ValueError, TypeError):
#                             pass
# 
#                         # try:
#                         #     pd.to_datetime(table_sample[column])
#                         #     continue
#                         # except:
#                         #     pass
# 
#                         if (
#                             table_sample[column]
#                             .apply(lambda x: isinstance(x, str))
#                             .all()
#                         ):
#                             continue
# 
#                 sample_tables[table_name] = table_sample
# 
#             return sample_tables
#         except psycopg2.Error as err:
#             raise ValueError(f"Failed to get summaries: {err}")
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

from ...src.mngs..db._PostgreSQLMixins._MaintenanceMixin import *

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
