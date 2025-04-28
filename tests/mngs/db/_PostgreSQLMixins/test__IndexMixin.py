# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/db/_PostgreSQLMixins/_IndexMixin.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-02-27 22:15:05 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_dev/src/mngs/db/_PostgreSQLMixins/_IndexMixin.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_PostgreSQLMixins/_IndexMixin.py"
# 
# from typing import List
# import psycopg2
# 
# class _IndexMixin:
#     def create_index(self, table_name: str, column_names: List[str],
#                     index_name: str = None, unique: bool = False) -> None:
#         try:
#             if index_name is None:
#                 index_name = f"idx_{table_name}_{'_'.join(column_names)}"
# 
#             unique_clause = "UNIQUE" if unique else ""
#             columns_str = ", ".join(column_names)
# 
#             query = f"""
#             CREATE {unique_clause} INDEX IF NOT EXISTS {index_name}
#             ON {table_name} ({columns_str})
#             """
#             self.execute(query)
# 
#         except (Exception, psycopg2.Error) as err:
#             raise ValueError(f"Failed to create index: {err}")
# 
#     def drop_index(self, index_name: str) -> None:
#         try:
#             self.execute(f"DROP INDEX IF EXISTS {index_name}")
#         except (Exception, psycopg2.Error) as err:
#             raise ValueError(f"Failed to drop index: {err}")
# 
#     def get_indexes(self, table_name: str = None) -> List[dict]:
#         try:
#             query = """
#             SELECT
#                 schemaname,
#                 tablename,
#                 indexname,
#                 indexdef
#             FROM
#                 pg_indexes
#             """
#             if table_name:
#                 query += f" WHERE tablename = '{table_name}'"
# 
#             return self.execute(query).fetchall()
# 
#         except (Exception, psycopg2.Error) as err:
#             raise ValueError(f"Failed to get indexes: {err}")
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/db/_PostgreSQLMixins/_IndexMixin.py
# --------------------------------------------------------------------------------
