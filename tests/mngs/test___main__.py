#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 07:12:50 (ywatanabe)"
# File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/mngs_repo/tests/mngs/test___main__.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/test___main__.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/__main__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-13 18:56:15 (ywatanabe)"
# # /home/yusukew/proj/mngs_repo/src/mngs/__main__.py
# 
# """
# 1. Functionality:
#    - Serves as the entry point for the mngs package
#    - Handles command-line arguments and directs to appropriate submodules
# 2. Input:
#    - Command-line arguments
# 3. Output:
#    - Execution of specified submodule or usage instructions
# 4. Prerequisites:
#    - mngs package and its submodules
# """
# 
# import warnings
# warnings.filterwarnings("ignore")
# 
# import sys
# from typing import List
# from .gen._print_config import print_config_main
# 
# def main(args: List[str] = None) -> None:
#     """
#     Main function to handle command-line arguments and execute appropriate submodules.
# 
#     Parameters
#     ----------
#     args : List[str], optional
#         Command-line arguments. If None, sys.argv[1:] will be used.
# 
#     Returns
#     -------
#     None
# 
#     Example
#     -------
#     >>> import mngs.__main__ as main_module
#     >>> main_module.main(['print_config', 'SOME_KEY'])
#     """
# 
# from .gen._print_config import print_config_main
# 
# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python -m mngs <command> [args]")
#         sys.exit(1)
# 
#     command = sys.argv[1]
#     args = sys.argv[2:]
# 
#     if command == "print_config":
#         print_config_main(args)
#     else:
#         print(f"Unknown command: {command}")
#         sys.exit(1)
# 
# if __name__ == "__main__":
#     main()

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/__main__.py
# --------------------------------------------------------------------------------


# def test_mngs_main():
#     try:
#         import mngs.__main__ as main_module

#         main_module.main(["print_config"])
#     except Exception as e:
#         print(e)
#         assert False


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# EOF