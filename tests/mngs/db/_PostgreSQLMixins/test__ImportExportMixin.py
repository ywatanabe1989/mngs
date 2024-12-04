# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-24 22:22:16 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/db/_PostgreSQL_modules/_ImportExportMixin.py
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_PostgreSQL_modules/_ImportExportMixin.py"
# 
# import pandas as pd
# from typing import List
# import psycopg2
# from io import StringIO
# 
# class _ImportExportMixin:
#     def load_from_csv(self, table_name: str, csv_path: str, if_exists: str = "append",
#                      batch_size: int = 10_000, chunk_size: int = 100_000) -> None:
#         with self.transaction():
#             try:
#                 if if_exists == "replace":
#                     self.execute(f"TRUNCATE TABLE {table_name}")
# 
#                 copy_sql = f"COPY {table_name} FROM STDIN WITH CSV HEADER"
#                 with open(csv_path, 'r') as f:
#                     self.cursor.copy_expert(sql=copy_sql, file=f)
# 
#             except (Exception, psycopg2.Error) as err:
#                 raise ValueError(f"Failed to import from CSV: {err}")
# 
#     def save_to_csv(self, table_name: str, output_path: str, columns: List[str] = ["*"],
#                     where: str = None, batch_size: int = 10_000) -> None:
#         try:
#             columns_str = ", ".join(columns) if columns != ["*"] else "*"
#             query = f"COPY (SELECT {columns_str} FROM {table_name}"
#             if where:
#                 query += f" WHERE {where}"
#             query += ") TO STDOUT WITH CSV HEADER"
# 
#             with open(output_path, 'w') as f:
#                 self.cursor.copy_expert(sql=query, file=f)
# 
#         except (Exception, psycopg2.Error) as err:
#             raise ValueError(f"Failed to export to CSV: {err}")
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

from mngs..db._PostgreSQLMixins._ImportExportMixin import *

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
