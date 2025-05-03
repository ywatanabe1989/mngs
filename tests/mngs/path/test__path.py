# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/path/_path.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 20:46:35 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/path/_path.py
# 
# import inspect
# 
# 
# def this_path(when_ipython="/tmp/fake.py"):
#     THIS_FILE = inspect.stack()[1].filename
#     if "ipython" in __file__:
#         THIS_FILE = when_ipython
#     return __file__
# 
# get_this_path = this_path
# 
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/path/_path.py
# --------------------------------------------------------------------------------
