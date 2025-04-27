# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_load_modules/_joblib.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:55:39 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_load_modules/_joblib.py
# 
# from typing import Any
# 
# import joblib
# 
# 
# def _load_joblib(lpath: str, **kwargs) -> Any:
#     """Load joblib file."""
#     if not lpath.endswith(".joblib"):
#         raise ValueError("File must have .joblib extension")
#     with open(lpath, "rb") as f:
#         return joblib.load(f, **kwargs)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_load_modules/_joblib.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
