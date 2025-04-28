# Add your tests here

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/_sh.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-03 02:23:16 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/_sh.py
# 
# import subprocess
# import mngs
# 
# def sh(command_str, verbose=True):
#     """
#     Executes a shell command from Python.
# 
#     Parameters:
#     - command_str (str): The command string to execute.
# 
#     Returns:
#     - output (str): The standard output from the executed command.
#     """
#     if verbose:
#         print(mngs.gen.color_text(f"{command_str}", "yellow"))
# 
#     process = subprocess.Popen(
#         command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
#     )
#     output, error = process.communicate()
#     if process.returncode == 0:
#         out = output.decode("utf-8").strip()
#     else:
#         out = error.decode("utf-8").strip()
# 
#     if verbose:
#         print(out)
# 
#     return out
# 
# 
# if __name__ == "__main__":
#     import sys
# 
#     import matplotlib.pyplot as plt
#     import mngs
# 
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False
#     )
#     sh("ls")
#     mngs.gen.close(CONFIG, verbose=False, notify=False)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/_sh.py
# --------------------------------------------------------------------------------
