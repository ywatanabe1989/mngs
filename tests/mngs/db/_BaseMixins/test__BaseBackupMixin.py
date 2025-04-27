# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/db/_BaseMixins/_BaseBackupMixin.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-02-27 22:16:38 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_dev/src/mngs/db/_BaseMixins/_BaseBackupMixin.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_BaseMixins/_BaseBackupMixin.py"
# 
# from typing import Optional
# 
# class _BaseBackupMixin:
#     def backup_table(self, table: str, file_path: str):
#         raise NotImplementedError
# 
#     def restore_table(self, table: str, file_path: str):
#         raise NotImplementedError
# 
#     def backup_database(self, file_path: str):
#         raise NotImplementedError
# 
#     def restore_database(self, file_path: str):
#         raise NotImplementedError
# 
#     def copy_table(self, source_table: str, target_table: str, where: Optional[str] = None):
#         raise NotImplementedError
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/db/_BaseMixins/_BaseBackupMixin.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
