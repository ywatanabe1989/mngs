# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/db/_BaseMixins/_BaseConnectionMixin.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-25 06:02:43 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/db/_BaseMixins/_BaseConnectionMixin.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_BaseMixins/_BaseConnectionMixin.py"
# 
# import threading
# from typing import Optional
# 
# class _BaseConnectionMixin:
#     def __init__(self):
#         self.lock = threading.Lock()
#         self._maintenance_lock = threading.Lock()
#         self.conn = None
#         self.cursor = None
# 
#     def __enter__(self):
#         return self
# 
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.close()
# 
#     def connect(self):
#         raise NotImplementedError
# 
#     def close(self):
#         raise NotImplementedError
# 
#     def reconnect(self):
#         raise NotImplementedError
# 
#     def execute(self, query: str, parameters = ()) -> None:
#         raise NotImplementedError
# 
#     def executemany(self, query: str, parameters) -> None:
#         raise NotImplementedError
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/db/_BaseMixins/_BaseConnectionMixin.py
# --------------------------------------------------------------------------------
