#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-20 18:02:40 (ywatanabe)"
# /home/ywatanabe/proj/mngs_repo/src/mngs/io/_cache.py

import os
from pathlib import Path

import mngs


def cache(
    dict_data=None,
    path=os.path.expanduser("~/.mngs/cache/recent.pkl"),
    id=None,
):
    """
    Store or fetch data (dict) employing a pickle file.

    Example:
        # Saving
        data = {"A": [1, 2, 3], "B": "example"}
        mngs.io.cache(data)

        # Loading
        cached_data = mngs.io.cache()

        # Checking the consistency
        assert data == cached_data

        # Using custom id
        mngs.io.cache({"score": 95}, id="ywata")
        ywata_data = mngs.io.cache(id="ywata")

    Parameters
    ----------
    dict_data : dict or None, optional
        Data dictionary for caching. If None, fetches stored data.
    path : str, optional
        Cache file location. Default: "~/.mngs/cache/recent.pkl".
    id : str or None, optional
        Unique identifier for cache file.

    Returns
    -------
    dict or None
        Retrieved data when dict_data is None, None otherwise.

    Raises
    ------
    TypeError
        When dict_data is neither a dictionary nor None.
    """
    cache_path = Path(path)

    if id:
        cache_path = cache_path.with_name(f"{id}.pkl")

    if isinstance(dict_data, dict):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        mngs.io.save(dict_data, str(cache_path))
    elif dict_data is None:
        return mngs.io.load(str(cache_path))
    else:
        raise TypeError("Input must be either a dictionary or None")
