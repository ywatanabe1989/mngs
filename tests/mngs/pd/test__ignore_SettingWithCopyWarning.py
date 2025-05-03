# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/pd/_ignore_SettingWithCopyWarning.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-05 07:35:30 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/pd/_ignore_.py
# 
# def ignore_SettingWithCopyWarning():
#     import warnings
#     try:
#         from pandas.errors import SettingWithCopyWarning
#     except:
#         from pandas.core.common import SettingWithCopyWarning
#     warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
#     # return SettingWithCopyWarning
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/pd/_ignore_SettingWithCopyWarning.py
# --------------------------------------------------------------------------------
