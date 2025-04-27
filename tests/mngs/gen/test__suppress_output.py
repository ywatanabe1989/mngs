# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/_suppress_output.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-07 19:30:31 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/gen/_suppress_output.py
# 
# import os
# from contextlib import contextmanager, redirect_stderr, redirect_stdout
# 
# 
# @contextmanager
# def suppress_output(suppress=True):
#     """
#     A context manager that suppresses stdout and stderr.
# 
#     Example:
#         with suppress_output():
#             print("This will not be printed to the console.")
#     """
#     if suppress:
#         # Open a file descriptor that points to os.devnull (a black hole for data)
#         with open(os.devnull, "w") as fnull:
#             # Temporarily redirect stdout and stderr to the file descriptor fnull
#             with redirect_stdout(fnull), redirect_stderr(fnull):
#                 # Yield control back to the context block
#                 yield
#     else:
#         # If suppress is False, just yield without redirecting output
#         yield
# 
# 
# quiet = suppress_output
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/_suppress_output.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
