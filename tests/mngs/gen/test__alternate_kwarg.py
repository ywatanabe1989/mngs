# Add your tests here

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/gen/_alternate_kwarg.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 13:30:41 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/gen/_alternate_kwarg.py
# 
# 
# def alternate_kwarg(kwargs, primary_key, alternate_key):
#     alternate_value = kwargs.pop(alternate_key, None)
#     kwargs[primary_key] = kwargs.get(primary_key) or alternate_value
#     return kwargs
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/gen/_alternate_kwarg.py
# --------------------------------------------------------------------------------
