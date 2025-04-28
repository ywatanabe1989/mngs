# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/db/_BaseMixins/_BaseImportExportMixin.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-24 22:20:15 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/db/_Basemodules/_BaseImportExportMixin.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_Basemodules/_BaseImportExportMixin.py"
# 
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# 
# from typing import List
# 
# class _BaseImportExportMixin:
#     def load_from_csv(self, table_name: str, csv_path: str, if_exists: str = "append",
#                      batch_size: int = 10_000, chunk_size: int = 100_000) -> None:
#         raise NotImplementedError
# 
#     def save_to_csv(self, table_name: str, output_path: str, columns: List[str] = ["*"],
#                    where: str = None, batch_size: int = 10_000) -> None:
#         raise NotImplementedError
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/db/_BaseMixins/_BaseImportExportMixin.py
# --------------------------------------------------------------------------------
