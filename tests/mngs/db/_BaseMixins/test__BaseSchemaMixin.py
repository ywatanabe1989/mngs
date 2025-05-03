# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/db/_BaseMixins/_BaseSchemaMixin.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-24 22:14:24 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/db/_Basemodules/_BaseSchemaMixin.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_Basemodules/_BaseSchemaMixin.py"
# 
# from typing import List, Dict, Any, Optional
# 
# class _BaseSchemaMixin:
#     def get_tables(self) -> List[str]:
#         raise NotImplementedError
# 
#     def get_columns(self, table: str) -> List[Dict[str, Any]]:
#         raise NotImplementedError
# 
#     def get_primary_keys(self, table: str) -> List[str]:
#         raise NotImplementedError
# 
#     def get_foreign_keys(self, table: str) -> List[Dict[str, Any]]:
#         raise NotImplementedError
# 
#     def get_indexes(self, table: str) -> List[Dict[str, Any]]:
#         raise NotImplementedError
# 
#     def table_exists(self, table: str) -> bool:
#         raise NotImplementedError
# 
#     def column_exists(self, table: str, column: str) -> bool:
#         raise NotImplementedError
# 
#     def create_index(self, table: str, columns: List[str], index_name: Optional[str] = None) -> None:
#         raise NotImplementedError
# 
#     def drop_index(self, index_name: str) -> None:
#         raise NotImplementedError
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/db/_BaseMixins/_BaseSchemaMixin.py
# --------------------------------------------------------------------------------
