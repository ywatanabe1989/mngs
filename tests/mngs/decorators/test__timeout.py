# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/decorators/_timeout.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-07 05:58:41 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/decorators/_timeout.py
# 
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-23 19:11:33"
# # Author: Yusuke Watanabe (ywata1989@gmail.com)
# 
# """
# This script does XYZ.
# """
# 
# """
# Imports
# """
# 
# 
# """
# Config
# """
# # CONFIG = mngs.gen.load_configs()
# 
# """
# Functions & Classes
# """
# from multiprocessing import Process, Queue
# 
# 
# def timeout(seconds=10, error_message="Timeout"):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             def queue_wrapper(queue, args, kwargs):
#                 result = func(*args, **kwargs)
#                 queue.put(result)
# 
#             queue = Queue()
#             args_for_process = (queue, args, kwargs)
#             process = Process(target=queue_wrapper, args=args_for_process)
#             process.start()
#             process.join(timeout=seconds)
# 
#             if process.is_alive():
#                 process.terminate()
#                 raise TimeoutError(error_message)
#             else:
#                 return queue.get()
# 
#         return wrapper
# 
#     return decorator
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/decorators/_timeout.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
