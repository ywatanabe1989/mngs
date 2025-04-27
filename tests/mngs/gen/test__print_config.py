# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/_print_config.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-13 18:53:04 (ywatanabe)"
# # /home/yusukew/proj/mngs_repo/src/mngs/gen/_print_config.py
# 
# """
# 1. Functionality:
#    - Prints configuration values from YAML files
# 2. Input:
#    - Configuration key (dot-separated for nested structures)
# 3. Output:
#    - Corresponding configuration value
# 4. Prerequisites:
#    - mngs package with load_configs function
# 
# Example:
#     python _print_config.py PATH.TITAN.MAT
# """
# 
# 
# import sys
# import os
# import argparse
# from pprint import pprint
# import sys
# 
# 
# def print_config(key):
# 
#     CONFIG = mngs.io.load_configs()
# 
#     if key is None:
#         print("Available configurations:")
#         pprint(CONFIG)
#         return
# 
#     try:
#         keys = key.split(".")
#         value = CONFIG
#         for k in keys:
# 
#             if isinstance(value, (dict, mngs.gen.utils._DotDict.DotDict)):
#                 value = value.get(k)
# 
#             elif isinstance(value, list):
#                 try:
#                     value = value[int(k)]
#                 except (ValueError, IndexError):
#                     value = None
# 
#             elif isinstance(value, str):
#                 break
# 
#             else:
#                 value = None
# 
#             if value is None:
#                 break
# 
#         print(value)
# 
#     except Exception as e:
#         print(f"Error: {e}")
#         print("Available configurations:")
#         pprint(value)
# 
# 
# def print_config_main(args=None):
#     if args is None:
#         args = sys.argv[1:]
# 
#     parser = argparse.ArgumentParser(description="Print configuration values")
#     parser.add_argument(
#         "key",
#         nargs="?",
#         default=None,
#         help="Configuration key (dot-separated for nested structures)",
#     )
#     parsed_args = parser.parse_args(args)
#     print_config(parsed_args.key)
# 
# 
# if __name__ == "__main__":
#     print_config_main()

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/_print_config.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
