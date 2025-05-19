# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dict/_to_str.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "ywatanabe (2024-11-03 00:48:22)"
# # File: ./mngs_repo/src/mngs/dict/_to_str.py
# 
# 
# def to_str(dictionary, delimiter="_"):
#     """
#     Convert a dictionary to a string representation.
# 
#     Example
#     -------
#     input_dict = {'a': 1, 'b': 2, 'c': 3}
#     result = dict2str(input_dict)
#     print(result)  # Output: a-1_b-2_c-3
# 
#     Parameters
#     ----------
#     dictionary : dict
#         The input dictionary to be converted.
#     delimiter : str, optional
#         The separator between key-value pairs (default is "_").
# 
#     Returns
#     -------
#     str
#         A string representation of the input dictionary.
#     """
#     return delimiter.join(
#         f"{key}-{value}" for key, value in dictionary.items()
#     )
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/dict/_to_str.py
# --------------------------------------------------------------------------------
