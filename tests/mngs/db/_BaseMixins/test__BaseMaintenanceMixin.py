# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/db/_BaseMixins/_BaseMaintenanceMixin.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-24 22:12:07 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/db/_Basemodules/_BaseMaintenanceMixin.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_Basemodules/_BaseMaintenanceMixin.py"
# 
# from typing import Optional, List, Dict
# 
# class _BaseMaintenanceMixin:
#     def vacuum(self, table: Optional[str] = None):
#         raise NotImplementedError
# 
#     def analyze(self, table: Optional[str] = None):
#         raise NotImplementedError
# 
#     def reindex(self, table: Optional[str] = None):
#         raise NotImplementedError
# 
#     def get_table_size(self, table: str):
#         raise NotImplementedError
# 
#     def get_database_size(self):
#         raise NotImplementedError
# 
#     def get_table_info(self) -> List[Dict]:
#         raise NotImplementedError
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/db/_BaseMixins/_BaseMaintenanceMixin.py
# --------------------------------------------------------------------------------
