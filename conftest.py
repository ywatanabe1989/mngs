#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 00:18:40 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/mngs_repo/conftest.py
# ----------------------------------------
import os
__FILE__ = (
    "./conftest.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
from pathlib import Path


def pytest_collect_file(file_path):
    # Only load files that have test functions
    if str(file_path).endswith(".py") and (
        file_path.name.startswith("test_")
        or file_path.name.endswith("_test.py")
    ):
        try:
            content = Path(file_path).read_text()
            if "def test_" not in content:
                return None
        except:
            pass
    return None


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-28 00:09:36 (ywatanabe)"
# # File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/mngs_repo/conftest.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./conftest.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------

# from pathlib import Path


# def pytest_collect_file(file_path, path, parent):
#     if str(file_path).endswith(".py") and (
#         file_path.name.startswith("test_")
#         or file_path.name.endswith("_test.py")
#     ):
#         # Check if file contains test functions before collection
#         try:
#             content = Path(file_path).read_text()
#             if "def test_" not in content:
#                 return None
#         except:
#             pass
#     return None  # Let pytest use its default collector

# # EOF

# EOF