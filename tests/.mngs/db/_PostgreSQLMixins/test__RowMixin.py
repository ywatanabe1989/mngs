# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-24 22:23:56 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/db/_PostgreSQL_modules/_RowMixin.py
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_PostgreSQL_modules/_RowMixin.py"
# 
# from typing import List, Optional
# import pandas as pd
# import psycopg2
# 
# class _RowMixin:
#     def get_rows(self, table_name: str, columns: List[str] = None,
#                  where: str = None, order_by: str = None,
#                  limit: Optional[int] = None, offset: Optional[int] = None,
#                  return_as: str = "dataframe"):
#         try:
#             if columns is None:
#                 columns_str = "*"
#             elif isinstance(columns, str):
#                 columns_str = f'"{columns}"'
#             else:
#                 columns_str = ", ".join(f'"{col}"' for col in columns)
# 
#             query_parts = [f"SELECT {columns_str} FROM {table_name}"]
# 
#             if where:
#                 query_parts.append(f"WHERE {where}")
#             if order_by:
#                 query_parts.append(f"ORDER BY {order_by}")
#             if limit is not None:
#                 query_parts.append(f"LIMIT {limit}")
#             if offset is not None:
#                 query_parts.append(f"OFFSET {offset}")
# 
#             query = " ".join(query_parts)
#             self.cursor.execute(query)
# 
#             column_names = [desc[0] for desc in self.cursor.description]
#             data = self.cursor.fetchall()
# 
#             if return_as == "list":
#                 return data
#             elif return_as == "dict":
#                 return [dict(zip(column_names, row)) for row in data]
#             else:
#                 return pd.DataFrame(data, columns=column_names)
# 
#         except (Exception, psycopg2.Error) as err:
#             raise ValueError(f"Query execution failed: {err}")
# 
#     def get_row_count(self, table_name: str = None, where: str = None) -> int:
#         try:
#             if table_name is None:
#                 raise ValueError("Table name must be specified")
# 
#             query = f"SELECT COUNT(*) FROM {table_name}"
#             if where:
#                 query += f" WHERE {where}"
# 
#             self.cursor.execute(query)
#             return self.cursor.fetchone()[0]
# 
#         except (Exception, psycopg2.Error) as err:
#             raise ValueError(f"Failed to get row count: {err}")
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

from mngs..db._PostgreSQLMixins._RowMixin import *

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
