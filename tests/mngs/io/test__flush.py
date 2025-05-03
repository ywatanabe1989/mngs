# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/io/_flush.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 03:23:44 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/io/_flush.py
# 
# import os
# import sys
# import warnings
# 
# 
# def flush(sys=sys):
#     """
#     Flushes the system's stdout and stderr, and syncs the file system.
#     This ensures all pending write operations are completed.
#     """
#     if sys is None:
#         warnings.warn("flush needs sys. Skipping.")
#     else:
#         sys.stdout.flush()
#         sys.stderr.flush()
#         os.sync()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/io/_flush.py
# --------------------------------------------------------------------------------
