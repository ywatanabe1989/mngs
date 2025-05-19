# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/_title2path.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: 2024-05-12 21:02:21 (7)
# # /sshx:ywatanabe@444:/home/ywatanabe/proj/mngs/src/mngs/gen/_title2spath.py
# 
# 
# def title2path(title):
#     """
#     Convert a title (string or dictionary) to a path-friendly string.
# 
#     Parameters
#     ----------
#     title : str or dict
#         The input title to be converted.
# 
#     Returns
#     -------
#     str
#         A path-friendly string derived from the input title.
#     """
#     if isinstance(title, dict):
#         from mngs.gen import dict2str
# 
#         title = dict2str(title)
# 
#     path = title
# 
#     patterns = [":", ";", "=", "[", "]"]
#     for pattern in patterns:
#         path = path.replace(pattern, "")
# 
#     path = path.replace("_-_", "-")
#     path = path.replace(" ", "_")
# 
#     while "__" in path:
#         path = path.replace("__", "_")
# 
#     return path.lower()
# 
# 
# # def title2path(title):
# #     if isinstance(title, dict):
# #         title = dict2str(title)
# 
# #     path = title
# 
# #     # Comma patterns
# #     patterns = [":", ";", "=", "[", "]"]
# #     for pp in patterns:
# #         path = path.replace(pp, "")
# 
# #     # Exceptions
# #     path = path.replace("_-_", "-")
# #     path = path.replace(" ", "_")
# 
# #     # Consective under scores
# #     for _ in range(10):
# #         path = path.replace("__", "_")
# 
# #     return path.lower()

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/_title2path.py
# --------------------------------------------------------------------------------
