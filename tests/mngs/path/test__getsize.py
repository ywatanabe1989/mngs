# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/path/_getsize.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 19:54:02 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/path/_getsize.py
# 
# import os
# 
# import numpy as np
# 
# 
# def getsize(path):
#     if os.path.exists(path):
#         return os.path.getsize(path)
#     else:
#         return np.nan
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/path/_getsize.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
