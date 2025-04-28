# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/str/_remove_ansi.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 01:21:34 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/str/_remove_ansi.py
# 
# import re
# 
# def remove_ansi(string):
#     """
#     Removes ANSI escape sequences from a given text chunk.
# 
#     Parameters:
#     - chunk (str): The text chunk to be cleaned.
# 
#     Returns:
#     - str: The cleaned text chunk.
#     """
#     ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
#     return ansi_escape.sub("", string)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/str/_remove_ansi.py
# --------------------------------------------------------------------------------
