#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-20 19:42:38 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/io/_cache.py


import os
import pickle
import sys
from pathlib import Path


# Working
def cache(id, *args):
    """
    Store or fetch data using a pickle file.

    Example:
        import mngs
        import numpy as np

        # Variables to cache
        var1 = "x"
        var2 = 1
        var3 = np.ones(10)

        # Saving
        var1, var2, var3 = mngs.io.cache("my_id", "var1", "var2", "var3")
        print(var1, var2, var3)

        # Loading when not all variables are defined and the id exists
        del var1, var2, var3
        var1, var2, var3 = mngs.io.cache("my_id", "var1", "var2", "var3")
        print(var1, var2, var3)


    Parameters
    ----------
    id : str
        Unique identifier for cache file.
    *args : str
        Variable names to be cached or loaded.

    Returns
    -------
    tuple
        Tuple of cached values corresponding to the input variable names.
    """
    cache_dir = Path.home() / ".cache" / "your_app_name"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{id}.pkl"

    does_cache_file_exist = cache_file.exists()

    # Get the caller's local variables
    caller_locals = sys._getframe(1).f_locals
    are_all_variables_defined = all(arg in caller_locals for arg in args)

    if are_all_variables_defined:
        # If all variables are defined, save them to cache and return as-is
        data_to_cache = {arg: caller_locals[arg] for arg in args}
        with cache_file.open("wb") as f:
            pickle.dump(data_to_cache, f)
        return tuple(data_to_cache.values())
    else:
        if does_cache_file_exist:
            # If cache exists, load and return the values
            with cache_file.open("rb") as f:
                loaded_data = pickle.load(f)
            return tuple(loaded_data[arg] for arg in args)
        else:
            raise ValueError(
                "Cache file not found and not all variables are defined."
            )


# Usage example
if __name__ == "__main__":
    import mngs
    import numpy as np

    # Variables to cache
    var1 = "x"
    var2 = 1
    var3 = np.ones(10)

    # Saving
    var1, var2, var3 = mngs.io.cache("my_id", "var1", "var2", "var3")
    print(var1, var2, var3)

    # Loading when not all variables are defined and the id exists
    del var1, var2, var3
    var1, var2, var3 = mngs.io.cache("my_id", "var1", "var2", "var3")
    print(var1, var2, var3)

    # var2 += 1
    # var1, var2, var3 = mngs.io.cache("my_id", "var1", "var2", "var3")
    # print(var1, var2, var3)

    # Working
    # # # Saving
    # # x = "test"
    # # y = 42
    # # z = np.array([1, 2, 8])
    # # x, y, z = cache("example", "x", "y", "z")
    # # # Loading
    # # x, y, z = cache("example", "x", "y", "z")
    # # print(f"Loaded x: {x}")
    # # print(f"Loaded y: {y}")
    # # print(f"Loaded z: {z}")
    # # Loading
    # # del x, y, z
    # x, y, z = mngs.io.cache("example", "x", "y", "z")
    # print(f"Loaded x: {x}")
    # print(f"Loaded y: {y}")
    # print(f"Loaded z: {z}")


# import os
# import sys
# from pathlib import Path

# import mngs


# def cache(id, *args):
#     """
#     Store or fetch data employing a pickle file.

#     Example:
#         import mngs
#         import numpy as np

#         # Variables to cache
#         var1 = "x"
#         var2 = 1
#         var3 = np.ones(10)

#         # Saving
#         var1, var2, var3 = mngs.io.cache("my_id", ["var1", "var2", "var3"])

#         # Loading when not all variables are defined and the id exists
#         del var1, var2, var3
#         var1, var2, var3 = mngs.io.cache("my_id", ["var1", "var2", "var3"])

#     Parameters
#     ----------
#     id : str
#         Unique identifier for cache file.
#     args : list
#         List of variable names to be cached or loaded.

#     Returns
#     -------
#     tuple
#         Tuple of cached values corresponding to the input variable names.
#     """
#     path = os.path.expanduser(f"~/.mngs/cache/{str(id)}.pkl")
#     caller_globals = sys._getframe(1).f_globals
#     vars_defined = [arg in caller_globals for arg in args]

#     if all(vars_defined):
#         data_to_cache = {arg: caller_globals[arg] for arg in args}
#         mngs.io.save(data_to_cache, path)
#         return tuple(data_to_cache.values())
#     elif os.path.exists(path):
#         cached = mngs.io.load(path)
#         for arg in args:
#             caller_globals[arg] = cached[arg]
#         return tuple(cached[arg] for arg in args)
#     else:
#         raise ValueError("Cache file not found and variables are not defined.")


# # def cache(id, args):
# #     """
# #     Store or fetch data employing a pickle file.

# #     Example:
# #         import mngs
# #         import numpy as np

# #         ########################################
# #         # Saving when all variables are defined
# #         ########################################
# #         # Variables to cache
# #         var1 = "x"
# #         var2 = 1
# #         var3 = np.ones(10)

# #         var1, var2, var3 = mngs.io.cache("my_id", ["var1", "var2", "var3"])

# #         ########################################
# #         # Loading when either of var1, var2, or var3 are not defined and id exists
# #         ########################################
# #         print(var1)
# #         # In [1]:         print(var1)
# #         # ---------------------------------------------------------------------------
# #         # NameError                                 Traceback (most recent call last)
# #         # Cell In[1], line 1
# #         # ----> 1 print(var1)

# #         # NameError: name 'var1' is not defined

# #         # In [2]:         var1, var2, var3 = mngs.io.cache("my_id", ["var1", "var2", "var3"])

# #         # In [3]:         print(var1)
# #         # x

# #     Parameters
# #     ----------
# #     id : str
# #         Unique identifier for cache file.
# #     args : list
# #         List of variable names to be cached or loaded.

# #     Returns
# #     -------
# #     tuple
# #         Tuple of cached values corresponding to the input variable names.
# #     """
# #     path = os.path.expanduser(f"~/.mngs/cache/{str(id)}.pkl")

# #     vars_defined = [arg in globals() for arg in args]

# #     if all(vars_defined):
# #         data_to_cache = {arg: globals()[arg] for arg in args}
# #         mngs.io.save(data_to_cache, path)
# #         return tuple(data_to_cache.values())
# #     elif os.path.exists(path):
# #         cached = mngs.io.load(path)
# #         for arg in args:
# #             globals()[arg] = cached[arg]
# #         return tuple(cached[arg] for arg in args)
# #     else:
# #         raise ValueError(
# #             f"Cache file not found and variables {args} are not defined."
# #         )


# if __name__ == "__main__":
#     import mngs
#     import numpy as np

#     # Variables to cache
#     var1 = "x"
#     var2 = 1
#     var3 = np.ones(10)

#     # Saving
#     var1, var2, var3 = mngs.io.cache("my_id", ["var1", "var2", "var3"])

#     # Loading
#     del var1, var2, var3
#     var1, var2, var3 = mngs.io.cache("my_id", ["var1", "var2", "var3"])


# # def _cache(
# #     dict_data=None,
# #     path=os.path.expanduser("~/.mngs/cache/recent.pkl"),
# #     id=None,
# # ):
# #     """
# #     Store or fetch data (dict) employing a pickle file.

# #     Example:
# #         # Saving
# #         data = {"A": [1, 2, 3], "B": "example"}
# #         mngs.io.cache(data)

# #         # Loading
# #         cached_data = mngs.io.cache()

# #         # Checking the consistency
# #         assert data == cached_data

# #         # Using custom id
# #         mngs.io.cache({"score": 95}, id="ywata")
# #         ywata_data = mngs.io.cache(id="ywata")

# #     Parameters
# #     ----------
# #     dict_data : dict or None, optional
# #         Data dictionary for caching. If None, fetches stored data.
# #     path : str, optional
# #         Cache file location. Default: "~/.mngs/cache/recent.pkl".
# #     id : str or None, optional
# #         Unique identifier for cache file.

# #     Returns
# #     -------
# #     dict or None
# #         Retrieved data when dict_data is None, None otherwise.

# #     Raises
# #     ------
# #     TypeError
# #         When dict_data is neither a dictionary nor None.
# #     """
# #     cache_path = Path(path)

# #     if id:
# #         cache_path = cache_path.with_name(f"{id}.pkl")

# #     if isinstance(dict_data, dict):
# #         cache_path.parent.mkdir(parents=True, exist_ok=True)
# #         mngs.io.save(dict_data, str(cache_path))
# #     elif dict_data is None:
# #         return mngs.io.load(str(cache_path))
# #     else:
# #         raise TypeError("Input must be either a dictionary or None")
