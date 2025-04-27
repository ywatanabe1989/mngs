# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/decorators/_not_implemented.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-07 22:16:25 (ywatanabe)"
# # /home/ywatanabe/proj/mngs/src/mngs/gen/_not_implemented.py
# 
# import warnings
# 
# 
# def not_implemented(func):
#     """
#     Decorator to mark methods as not implemented, issue a warning, and prevent their execution.
# 
#     Arguments:
#         func (callable): The function or method to decorate.
# 
#     Returns:
#         callable: A wrapper function that issues a warning and raises NotImplementedError when called.
#     """
# 
#     def wrapper(*args, **kwargs):
#         # Issue a warning before raising the error
#         warnings.warn(
#             f"Attempt to use unimplemented method: '{func.__name__}'. This method is not yet available.",
#             category=FutureWarning,
#             stacklevel=2,
#         )
#         # # Raise the NotImplementedError
#         # raise NotImplementedError(f"The method '{func.__name__}' is not implemented yet.")
# 
#     return wrapper

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/decorators/_not_implemented.py
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])
