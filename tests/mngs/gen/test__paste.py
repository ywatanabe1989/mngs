# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/gen/_paste.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-03 02:13:54 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/gen/_paste.py
# def paste():
#     import textwrap
# 
#     import pyperclip
# 
#     try:
#         clipboard_content = pyperclip.paste()
#         clipboard_content = textwrap.dedent(clipboard_content)
#         exec(clipboard_content)
#     except Exception as e:
#         print(f"Could not execute clipboard content: {e}")
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/gen/_paste.py
# --------------------------------------------------------------------------------
