# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/reproduce/_gen_ID.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-07 17:53:38 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/reproduce/_gen_ID.py
# 
# import random as _random
# import string as _string
# from datetime import datetime as _datetime
# 
# 
# def gen_ID(time_format="%YY-%mM-%dD-%Hh%Mm%Ss", N=8):
#     now_str = _datetime.now().strftime(time_format)
#     rand_str = "".join(
#         [_random.choice(_string.ascii_letters + _string.digits) for i in range(N)]
#     )
#     return now_str + "_" + rand_str
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/reproduce/_gen_ID.py
# --------------------------------------------------------------------------------
