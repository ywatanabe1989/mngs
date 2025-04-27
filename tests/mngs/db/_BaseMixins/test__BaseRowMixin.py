# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/db/_BaseMixins/_BaseRowMixin.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-24 22:21:03 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/db/_Basemodules/_BaseRowMixin.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_Basemodules/_BaseRowMixin.py"
# 
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# 
# from typing import List, Optional
# 
# class _BaseRowMixin:
#     def get_rows(self, table_name: str, columns: List[str] = None,
#                  where: str = None, order_by: str = None,
#                  limit: Optional[int] = None, offset: Optional[int] = None,
#                  return_as: str = "dataframe"):
#         raise NotImplementedError
# 
#     def get_row_count(self, table_name: str = None, where: str = None) -> int:
#         raise NotImplementedError
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/db/_BaseMixins/_BaseRowMixin.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
